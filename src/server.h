#ifndef LLM_SERVER_H
#define LLM_SERVER_H

// Minimal OpenAI-compatible HTTP server for llm.cpp.
// Implements:
//   GET  /           - HTML web chat interface
//   GET  /v1/models  - model list
//   POST /v1/chat/completions - chat completion (streaming + non-streaming)
// Features: API key authentication, tool/function calling, image message
// support (base64 URLs passed through to vision-capable models).
// Uses only POSIX sockets (Linux/macOS) or Winsock (Windows).
// No external dependencies required.

#include <cstdio>
#include <cstring>
#include <ctime>
#include <functional>
#include <string>
#include <vector>

#ifdef _WIN32
#  include <winsock2.h>
#  include <ws2tcpip.h>
#  pragma comment(lib, "ws2_32.lib")
   typedef SOCKET socket_t;
#  define CLOSE_SOCKET(s) closesocket(s)
#  define SOCK_INVALID INVALID_SOCKET
#else
#  include <sys/socket.h>
#  include <netinet/in.h>
#  include <unistd.h>
   typedef int socket_t;
#  define CLOSE_SOCKET(s) close(s)
#  define SOCK_INVALID    (-1)
#endif

#include "model.h"
#include "sampler.h"

// ---- Minimal JSON helpers ----


// Extract the string value of a JSON key. Returns "" if not found.
static std::string json_get_str(const std::string& json, const std::string& key) {
    std::string needle = "\"" + key + "\"";
    size_t kpos = json.find(needle);
    if (kpos == std::string::npos) return "";
    size_t colon = json.find(':', kpos + needle.size());
    if (colon == std::string::npos) return "";
    size_t q1 = json.find('"', colon + 1);
    if (q1 == std::string::npos) return "";
    size_t q2 = q1 + 1;
    while (q2 < json.size()) {
        if (json[q2] == '\\') { q2 += 2; continue; }
        if (json[q2] == '"')  { break; }
        q2++;
    }
    // Unescape common sequences
    std::string raw = json.substr(q1 + 1, q2 - q1 - 1);
    std::string out;
    out.reserve(raw.size());
    for (size_t i = 0; i < raw.size(); i++) {
        if (raw[i] == '\\' && i + 1 < raw.size()) {
            switch (raw[i+1]) {
                case 'n':  out += '\n'; i++; break;
                case 't':  out += '\t'; i++; break;
                case 'r':  out += '\r'; i++; break;
                case '"':  out += '"';  i++; break;
                case '\\': out += '\\'; i++; break;
                default:   out += raw[i+1]; i++; break;
            }
        } else {
            out += raw[i];
        }
    }
    return out;
}

// Extract a numeric value for a JSON key. Returns default_val if not found.
static double json_get_num(const std::string& json, const std::string& key,
                           double default_val) {
    std::string needle = "\"" + key + "\"";
    size_t kpos = json.find(needle);
    if (kpos == std::string::npos) return default_val;
    size_t colon = json.find(':', kpos + needle.size());
    if (colon == std::string::npos) return default_val;
    // skip whitespace
    size_t vpos = colon + 1;
    while (vpos < json.size() && (json[vpos] == ' ' || json[vpos] == '\t' ||
                                  json[vpos] == '\r' || json[vpos] == '\n'))
        vpos++;
    if (vpos >= json.size()) return default_val;
    char* end = nullptr;
    double v = strtod(json.c_str() + vpos, &end);
    return (end > json.c_str() + vpos) ? v : default_val;
}

// Extract a bool value for a JSON key. Returns default_val if not found.
static bool json_get_bool(const std::string& json, const std::string& key,
                          bool default_val) {
    std::string needle = "\"" + key + "\"";
    size_t kpos = json.find(needle);
    if (kpos == std::string::npos) return default_val;
    size_t colon = json.find(':', kpos + needle.size());
    if (colon == std::string::npos) return default_val;
    size_t vpos = colon + 1;
    while (vpos < json.size() && (json[vpos] == ' ' || json[vpos] == '\t' ||
                                  json[vpos] == '\r' || json[vpos] == '\n'))
        vpos++;
    if (vpos >= json.size()) return default_val;
    if (json.compare(vpos, 4, "true")  == 0) return true;
    if (json.compare(vpos, 5, "false") == 0) return false;
    return default_val;
}

// Escape a string for JSON output.
static std::string json_escape(const std::string& s) {
    std::string out;
    out.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  out += "\\\""; break;
            case '\\': out += "\\\\"; break;
            case '\n': out += "\\n";  break;
            case '\r': out += "\\r";  break;
            case '\t': out += "\\t";  break;
            default:
                if (c < 0x20) {
                    char buf[8];
                    snprintf(buf, sizeof(buf), "\\u%04x", c);
                    out += buf;
                } else {
                    out += static_cast<char>(c);
                }
        }
    }
    return out;
}

// ---- HTTP helpers ----

static std::string http_recv_all(socket_t fd) {
    std::string buf;
    buf.reserve(4096);
    char tmp[4096];
    while (true) {
#ifdef _WIN32
        int n = recv(fd, tmp, sizeof(tmp), 0);
#else
        ssize_t n = recv(fd, tmp, sizeof(tmp), 0);
#endif
        if (n <= 0) break;
        buf.append(tmp, static_cast<size_t>(n));
        // If we've seen the end of headers, check Content-Length
        size_t hdr_end = buf.find("\r\n\r\n");
        if (hdr_end == std::string::npos) continue;
        // Look for Content-Length header
        size_t cl_pos = buf.find("Content-Length:");
        if (cl_pos == std::string::npos) cl_pos = buf.find("content-length:");
        if (cl_pos != std::string::npos) {
            size_t val_start = buf.find_first_not_of(" \t", cl_pos + 15);
            size_t val_end   = buf.find("\r\n", val_start);
            if (val_end != std::string::npos) {
                int content_len = atoi(buf.c_str() + val_start);
                size_t body_start = hdr_end + 4;
                if (static_cast<int>(buf.size() - body_start) >= content_len) break;
                continue;
            }
        } else {
            break;  // No body expected
        }
    }
    return buf;
}

static void http_send(socket_t fd, const std::string& status,
                      const std::string& content_type,
                      const std::string& body) {
    std::string resp =
        "HTTP/1.1 " + status + "\r\n"
        "Content-Type: " + content_type + "\r\n"
        "Content-Length: " + std::to_string(body.size()) + "\r\n"
        "Access-Control-Allow-Origin: *\r\n"
        "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
        "Connection: close\r\n"
        "\r\n" + body;
    size_t sent = 0;
    while (sent < resp.size()) {
#ifdef _WIN32
        int n = send(fd, resp.c_str() + sent,
                     static_cast<int>(resp.size() - sent), 0);
#else
        ssize_t n = send(fd, resp.c_str() + sent, resp.size() - sent, 0);
#endif
        if (n <= 0) break;
        sent += static_cast<size_t>(n);
    }
}

// Send streaming Server-Sent Events (SSE) lines to client.
static void sse_send(socket_t fd, const std::string& data) {
    std::string line = "data: " + data + "\n\n";
#ifdef _WIN32
    send(fd, line.c_str(), static_cast<int>(line.size()), 0);
#else
    send(fd, line.c_str(), line.size(), 0);
#endif
}

// ---- Authorization helpers ----

// ASCII lower-case helper (avoids locale-dependent std::tolower).
static std::string str_to_lower(const std::string& s) {
    std::string out = s;
    for (char& c : out)
        c = (c >= 'A' && c <= 'Z') ? static_cast<char>(c + 32) : c;
    return out;
}

// Extract the value of a named HTTP header from a raw request string.
// Header name matching is case-insensitive.  Returns "" if not found.
static std::string http_get_header(const std::string& raw,
                                   const std::string& name) {
    std::string lname = str_to_lower(name) + ":";
    // Scan the raw request line-by-line (stop at blank line = header end)
    size_t pos = 0;
    while (pos < raw.size()) {
        size_t eol = raw.find("\r\n", pos);
        if (eol == std::string::npos) eol = raw.size();
        if (eol == pos) break;  // blank line
        std::string line = raw.substr(pos, eol - pos);
        if (str_to_lower(line.substr(0, lname.size())) == lname) {
            size_t vs = line.find_first_not_of(" \t", lname.size());
            if (vs != std::string::npos)
                return line.substr(vs);
        }
        pos = eol + 2;
    }
    return "";
}

// Return true if the request carries a valid Bearer token (or no key required).
static bool auth_check(const std::string& raw, const std::string& api_key) {
    if (api_key.empty()) return true;
    std::string auth = http_get_header(raw, "Authorization");
    // Accept "Bearer <key>" or just "<key>"
    if (auth.size() >= 7 && auth.substr(0, 7) == "Bearer ")
        auth = auth.substr(7);
    // Trim trailing whitespace
    while (!auth.empty() && (auth.back() == ' ' || auth.back() == '\r' ||
                              auth.back() == '\n'))
        auth.pop_back();
    return auth == api_key;
}

// ---- Tool-call helpers ----

// Detect and extract a <tool_call>JSON</tool_call> block from generated text.
// Returns the inner JSON string or "" if not found.
// Removes the block from `text` in-place.
static std::string extract_tool_call(std::string& text) {
    const std::string open_tag  = "<tool_call>";
    const std::string close_tag = "</tool_call>";
    size_t s = text.find(open_tag);
    if (s == std::string::npos) return "";
    size_t e = text.find(close_tag, s + open_tag.size());
    if (e == std::string::npos) return "";
    std::string inner = text.substr(s + open_tag.size(),
                                    e - s - open_tag.size());
    // Trim whitespace
    size_t i = inner.find_first_not_of(" \t\r\n");
    size_t j = inner.find_last_not_of(" \t\r\n");
    inner = (i == std::string::npos) ? "" : inner.substr(i, j - i + 1);
    // Remove the block from text
    text.erase(s, e + close_tag.size() - s);
    return inner;
}

// Build an OpenAI-compatible tool_calls JSON array string from a raw JSON
// object produced by the model (e.g. {"name":"fn","arguments":{...}}).
static std::string build_tool_calls_json(const std::string& call_json,
                                         long ts) {
    // Extract function name and arguments from the inner JSON
    std::string fn_name = json_get_str(call_json, "name");
    // Try "arguments" as a nested object (find raw value after key)
    std::string args_raw;
    {
        std::string needle = "\"arguments\"";
        size_t p = call_json.find(needle);
        if (p != std::string::npos) {
            size_t colon = call_json.find(':', p + needle.size());
            if (colon != std::string::npos) {
                size_t vs = call_json.find_first_not_of(" \t\r\n",
                                                        colon + 1);
                if (vs != std::string::npos) {
                    if (call_json[vs] == '{') {
                        // nested object - find matching }
                        int depth = 1; size_t ve = vs + 1;
                        while (ve < call_json.size() && depth > 0) {
                            if (call_json[ve] == '{') depth++;
                            else if (call_json[ve] == '}') depth--;
                            else if (call_json[ve] == '"') {
                                ve++;
                                while (ve < call_json.size() &&
                                       call_json[ve] != '"') {
                                    if (call_json[ve] == '\\') ve++;
                                    ve++;
                                }
                            }
                            ve++;
                        }
                        args_raw = call_json.substr(vs, ve - vs);
                    } else if (call_json[vs] == '"') {
                        // already a string
                        args_raw = json_get_str(call_json, "arguments");
                    }
                }
            }
        }
    }
    if (args_raw.empty()) args_raw = "{}";
    // Escape args_raw for embedding in a JSON string value
    std::string args_esc = json_escape(args_raw);
    std::string call_id = "call_" + std::to_string(ts);
    return "[{\"id\":\"" + call_id + "\","
           "\"type\":\"function\","
           "\"function\":{\"name\":\"" + json_escape(fn_name) + "\","
           "\"arguments\":\"" + args_esc + "\"}}]";
}

// ---- Image / multimodal message helpers ----

// Flatten a multimodal "content" array (list of {type, text/image_url} objects)
// to a plain text string.  Images are represented as "[image: <url>]".
// This allows vision models that accept the raw text token stream to process
// descriptions, while non-vision models receive a reasonable fallback.
static std::string flatten_content_array(const std::string& json,
                                         size_t arr_start) {
    // arr_start points to the '[' of the content array
    std::string result;
    size_t pos = arr_start + 1;
    int depth = 1;
    while (pos < json.size() && depth > 0) {
        if (json[pos] == '{') {
            size_t obj_s = pos;
            int od = 1; pos++;
            while (pos < json.size() && od > 0) {
                if (json[pos] == '{') od++;
                else if (json[pos] == '}') od--;
                else if (json[pos] == '"') {
                    pos++;
                    while (pos < json.size() && json[pos] != '"') {
                        if (json[pos] == '\\') pos++;
                        pos++;
                    }
                }
                pos++;
            }
            std::string obj = json.substr(obj_s, pos - obj_s);
            std::string type = json_get_str(obj, "type");
            if (type == "text") {
                if (!result.empty()) result += "\n";
                result += json_get_str(obj, "text");
            } else if (type == "image_url") {
                // Try nested image_url.url
                size_t iu = obj.find("\"image_url\"");
                std::string url;
                if (iu != std::string::npos) {
                    size_t brace = obj.find('{', iu);
                    if (brace != std::string::npos) {
                        int bd = 1; size_t be = brace + 1;
                        while (be < obj.size() && bd > 0) {
                            if (obj[be] == '{') bd++;
                            else if (obj[be] == '}') bd--;
                            else if (obj[be] == '"') {
                                be++;
                                while (be < obj.size() && obj[be] != '"') {
                                    if (obj[be] == '\\') be++;
                                    be++;
                                }
                            }
                            be++;
                        }
                        std::string iu_obj = obj.substr(brace, be - brace);
                        url = json_get_str(iu_obj, "url");
                    }
                }
                if (url.empty()) url = json_get_str(obj, "url");
                if (!result.empty()) result += "\n";
                // For base64 data URLs, just note an image was provided
                if (url.size() > 5 && url.substr(0, 5) == "data:") {
                    result += "[image attached]";
                } else {
                    result += "[image: " + url + "]";
                }
            }
        } else if (json[pos] == '[') {
            depth++; pos++;
        } else if (json[pos] == ']') {
            depth--; pos++;
        } else {
            pos++;
        }
    }
    return result;
}

// Parse the "messages" array, handling both plain string content and
// multimodal content arrays (vision-model format).
// Returns a vector of (role, content_text) pairs.
static std::vector<std::pair<std::string,std::string>>
json_parse_messages_multimodal(const std::string& json) {
    std::vector<std::pair<std::string,std::string>> msgs;
    size_t arr_start = json.find("\"messages\"");
    if (arr_start == std::string::npos) return msgs;
    size_t bracket = json.find('[', arr_start);
    if (bracket == std::string::npos) return msgs;
    size_t pos = bracket + 1;
    int depth = 1;
    while (pos < json.size() && depth > 0) {
        if (json[pos] == '{') {
            size_t obj_start = pos;
            int obj_depth = 1; pos++;
            while (pos < json.size() && obj_depth > 0) {
                if (json[pos] == '{') obj_depth++;
                else if (json[pos] == '}') obj_depth--;
                else if (json[pos] == '"') {
                    pos++;
                    while (pos < json.size() && json[pos] != '"') {
                        if (json[pos] == '\\') pos++;
                        pos++;
                    }
                }
                pos++;
            }
            std::string obj = json.substr(obj_start, pos - obj_start);
            std::string role = json_get_str(obj, "role");
            if (role.empty()) continue;

            // Determine content: string or array
            std::string content;
            // Look for "content": [  (array form)
            {
                std::string cneedle = "\"content\"";
                size_t cp = obj.find(cneedle);
                if (cp != std::string::npos) {
                    size_t colon = obj.find(':', cp + cneedle.size());
                    if (colon != std::string::npos) {
                        size_t vs = obj.find_first_not_of(" \t\r\n", colon + 1);
                        if (vs != std::string::npos && obj[vs] == '[') {
                            content = flatten_content_array(obj, vs);
                        } else {
                            content = json_get_str(obj, "content");
                        }
                    }
                }
            }
            msgs.push_back({role, content});
        } else if (json[pos] == '[') {
            depth++; pos++;
        } else if (json[pos] == ']') {
            depth--; pos++;
        } else {
            pos++;
        }
    }
    return msgs;
}

// ---- HTML web UI ----

// Returns the self-contained HTML chat interface.
// Supports i18n (English / Chinese / Japanese) and text file upload.
static std::string get_web_ui_html(int port) {
    (void)port;
    return R"HTML(<!DOCTYPE html>
<html lang="en" id="html-root">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title id="page-title">llm.cpp Chat</title>
<style>
:root{--bg:#0d1117;--surface:#161b22;--card:#21262d;--accent:#58a6ff;--danger:#f85149;--ok:#3fb950;--text:#e6edf3;--dim:#8b949e;--border:#30363d;--user:#1f3a5f;--radius:10px}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;background:var(--bg);color:var(--text);height:100vh;display:flex;overflow:hidden}
.sidebar{width:270px;background:var(--surface);border-right:1px solid var(--border);display:flex;flex-direction:column;flex-shrink:0}
.logo{padding:16px;border-bottom:1px solid var(--border)}
.logo h1{font-size:1.2rem;color:var(--accent);font-weight:700}
.logo p{font-size:.75rem;color:var(--dim);margin-top:3px}
.settings{padding:14px;flex:1;overflow-y:auto;display:flex;flex-direction:column;gap:14px}
.settings h3{font-size:.7rem;text-transform:uppercase;color:var(--dim);letter-spacing:.08em}
label{display:block;font-size:.78rem;color:var(--dim);margin-bottom:3px}
input[type=text],input[type=number],input[type=password],textarea,select{width:100%;background:var(--card);border:1px solid var(--border);color:var(--text);padding:7px 10px;border-radius:6px;font-size:.82rem;outline:none}
input[type=text]:focus,input[type=number]:focus,input[type=password]:focus,textarea:focus{border-color:var(--accent)}
textarea{resize:vertical;min-height:56px;font-family:inherit}
input[type=range]{width:100%;accent-color:var(--accent)}
.rv{color:var(--accent);font-size:.75rem;margin-left:6px}
.btn{background:var(--accent);color:#0d1117;border:none;padding:8px 16px;border-radius:var(--radius);cursor:pointer;font-size:.85rem;font-weight:600;transition:opacity .15s}
.btn:hover{opacity:.85}
.btn:disabled{opacity:.45;cursor:not-allowed}
.btn-sm{padding:5px 10px;font-size:.78rem}
.btn-ghost{background:var(--card);color:var(--text);border:1px solid var(--border)}
.btn-danger{background:var(--danger);color:#fff}
.btn-ok{background:var(--ok);color:#0d1117}
.main{flex:1;display:flex;flex-direction:column;overflow:hidden}
.messages{flex:1;overflow-y:auto;padding:20px;display:flex;flex-direction:column;gap:14px}
.msg{display:flex;gap:10px;max-width:88%}
.msg.user{align-self:flex-end;flex-direction:row-reverse}
.msg.assistant,.msg.system{align-self:flex-start}
.avatar{width:32px;height:32px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-size:.95rem;flex-shrink:0;margin-top:2px}
.user .avatar{background:#1f3a5f}
.assistant .avatar{background:var(--card)}
.bubble{background:var(--card);border-radius:var(--radius);padding:11px 15px;font-size:.88rem;line-height:1.65;max-width:100%;word-break:break-word}
.user .bubble{background:var(--user)}
pre{background:#010409;border-radius:6px;padding:12px;margin:8px 0;overflow-x:auto;font-size:.78rem;border:1px solid var(--border)}
code{font-family:"SF Mono",Consolas,monospace;background:rgba(110,118,129,.2);padding:1px 5px;border-radius:4px;font-size:.83em}
pre code{background:transparent;padding:0}
.tool-block{margin-top:8px;border:1px solid var(--border);border-radius:8px;padding:10px;background:rgba(0,0,0,.3)}
.tool-block h4{font-size:.72rem;color:var(--dim);margin-bottom:7px;text-transform:uppercase;letter-spacing:.06em}
.tool-item{background:var(--card);border-radius:6px;padding:8px 10px;margin-bottom:6px;font-size:.8rem}
.tool-item .fn{color:var(--accent);font-weight:600}
.tool-item .args{color:var(--dim);font-family:monospace;margin-top:3px;white-space:pre-wrap}
.tool-actions{display:flex;gap:6px;margin-top:7px}
.input-area{border-top:1px solid var(--border);padding:12px 16px}
.img-preview{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:6px}
.img-thumb{position:relative}
.img-thumb img{width:54px;height:54px;object-fit:cover;border-radius:6px;display:block}
.img-thumb .rm{position:absolute;top:-5px;right:-5px;background:var(--danger);color:#fff;border:none;width:17px;height:17px;border-radius:50%;cursor:pointer;font-size:.65rem;display:flex;align-items:center;justify-content:center;line-height:1}
.file-preview{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:8px}
.file-chip{display:flex;align-items:center;gap:5px;background:var(--card);border:1px solid var(--border);border-radius:6px;padding:4px 9px;font-size:.78rem;max-width:220px}
.file-chip span{overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:var(--accent)}
.file-chip .rm{background:transparent;color:var(--dim);border:none;cursor:pointer;font-size:.9rem;padding:0 0 0 4px;line-height:1;flex-shrink:0}
.file-chip .rm:hover{color:var(--danger)}
.input-row{display:flex;gap:8px;align-items:flex-end}
.input-row textarea{flex:1;background:var(--card);border:1px solid var(--border);color:var(--text);padding:9px 13px;border-radius:var(--radius);font-size:.88rem;resize:none;max-height:140px;overflow-y:auto;font-family:inherit;outline:none}
.input-row textarea:focus{border-color:var(--accent)}
.ic-btn{width:38px;height:38px;padding:0;border-radius:50%;display:flex;align-items:center;justify-content:center;flex-shrink:0;cursor:pointer;background:var(--card);border:1px solid var(--border);color:var(--text);font-size:1rem;transition:background .15s}
.ic-btn:hover{background:var(--border)}
.send-btn{background:var(--accent);color:#0d1117;border:none}
.send-btn:hover{opacity:.85}
.dots{display:inline-flex;gap:4px;align-items:center;padding:4px 0}
.dots span{width:7px;height:7px;background:var(--dim);border-radius:50%;animation:dot 1.2s infinite}
.dots span:nth-child(2){animation-delay:.2s}
.dots span:nth-child(3){animation-delay:.4s}
@keyframes dot{0%,80%,100%{transform:scale(.5);opacity:.4}40%{transform:scale(1);opacity:1}}
.toast{position:fixed;bottom:18px;right:18px;background:var(--card);color:var(--text);padding:10px 18px;border-radius:8px;border-left:4px solid var(--accent);font-size:.82rem;z-index:999;box-shadow:0 4px 16px rgba(0,0,0,.4);animation:fadein .2s}
@keyframes fadein{from{opacity:0;transform:translateY(8px)}to{opacity:1;transform:none}}
::-webkit-scrollbar{width:5px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:var(--border);border-radius:3px}
</style>
</head>
<body>
<div class="sidebar">
  <div class="logo"><h1>llm.cpp</h1><p data-i18n="subtitle">Lightweight LLM inference</p></div>
  <div class="settings">
    <div>
      <h3 data-i18n="language">Language</h3>
      <select id="lang" onchange="setLang(this.value)">
        <option value="en">English</option>
        <option value="zh">&#20013;&#25991;</option>
        <option value="ja">&#26085;&#26412;&#35486;</option>
      </select>
    </div>
    <div>
      <h3 data-i18n="sys_prompt_h">System Prompt</h3>
      <textarea id="sys" rows="3">You are a helpful assistant.</textarea>
    </div>
    <div>
      <label><span data-i18n="temperature">Temperature</span> <span class="rv" id="tv">0.8</span></label>
      <input type="range" id="temp" min="0" max="2" step="0.05" value="0.8" oninput="document.getElementById('tv').textContent=this.value">
    </div>
    <div>
      <label data-i18n="max_tokens">Max Tokens</label>
      <input type="number" id="maxt" value="512" min="1" max="8192">
    </div>
    <div>
      <label><span data-i18n="top_p">Top-P</span> <span class="rv" id="ppv">0.9</span></label>
      <input type="range" id="topp" min="0" max="1" step="0.01" value="0.9" oninput="document.getElementById('ppv').textContent=this.value">
    </div>
    <div>
      <label data-i18n="api_key_label">API Key (Bearer token)</label>
      <input type="password" id="apikey" data-i18n-ph="api_key_ph">
    </div>
    <div>
      <label data-i18n="server_url">Server URL</label>
      <input type="text" id="srvurl" value="">
    </div>
    <button class="btn btn-ghost btn-sm" style="width:100%" onclick="clearChat()" data-i18n="clear_chat">&#128465; Clear Chat</button>
    <div>
      <h3 data-i18n="agent_tools_h">Agent / Tools</h3>
      <label style="display:flex;align-items:center;gap:8px;cursor:pointer">
        <input type="checkbox" id="agent"> <span data-i18n="enable_tool">Enable tool calling</span>
      </label>
      <div style="font-size:.75rem;color:var(--dim);margin-top:6px" data-i18n="tool_demo">Built-in demo tools: calculator, get_datetime</div>
    </div>
  </div>
</div>
<div class="main">
  <div class="messages" id="msgs">
    <div class="msg assistant">
      <div class="avatar">&#129302;</div>
      <div class="bubble" id="greeting-bubble">Hello! I&#39;m powered by <strong>llm.cpp</strong>. How can I help you today?</div>
    </div>
  </div>
  <div class="input-area">
    <div class="img-preview" id="imgprev"></div>
    <div class="file-preview" id="fileprev"></div>
    <div class="input-row">
      <label class="ic-btn" data-i18n-title="attach_img" title="Attach image" style="cursor:pointer">
        &#128206;<input type="file" id="imgfile" accept="image/*" style="display:none" onchange="attachImg(this)">
      </label>
      <label class="ic-btn" data-i18n-title="attach_file" title="Attach file" style="cursor:pointer">
        &#128196;<input type="file" id="txtfile" accept=".txt,.md,.csv,.json,.xml,.py,.js,.ts,.cpp,.h,.c,.java,.go,.rs,.rb,.sh,.yaml,.yml,.toml,.ini,.log,.rst,.tex" style="display:none" onchange="attachFile(this)">
      </label>
      <textarea id="inp" rows="1" data-i18n-ph="msg_ph" placeholder="Message&#8230; (Enter to send, Shift+Enter for newline)"
        onkeydown="if(event.key==='Enter'&&!event.shiftKey){event.preventDefault();send();}autoH(this)"></textarea>
      <button class="ic-btn send-btn" id="sendbtn" onclick="send()" title="Send">&#10148;</button>
    </div>
  </div>
<script>
// ---- i18n translations ----
var LANG={
en:{
  title:'llm.cpp Chat',subtitle:'Lightweight LLM inference',language:'Language',
  sys_prompt_h:'System Prompt',temperature:'Temperature',max_tokens:'Max Tokens',
  top_p:'Top-P',api_key_label:'API Key (Bearer token)',api_key_ph:'Leave blank if not set',
  server_url:'Server URL',clear_chat:'\uD83D\uDDD1 Clear Chat',
  agent_tools_h:'Agent / Tools',enable_tool:'Enable tool calling',
  tool_demo:'Built-in demo tools: calculator, get_datetime',
  greeting:'Hello! I\'m powered by <strong>llm.cpp</strong>. How can I help you today?',
  msg_ph:'Message\u2026 (Enter to send, Shift+Enter for newline)',
  chat_cleared:'Chat cleared.',attach_img:'Attach image',attach_file:'Attach file',
  tool_calls_h:'\uD83D\uDD27 Tool Calls \u2014 confirm before execution',
  allow:'\u2713 Allow',deny:'\u2717 Deny',denied:'Denied',
  tool_not_impl:'Tool not implemented in demo.',
  file_prefix:'File: ',img_suffix:' image(s)'
},
zh:{
  title:'llm.cpp \u804A\u5929',subtitle:'\u8F7B\u91CF\u7EA7 LLM \u63A8\u7406\u5F15\u64CE',
  language:'\u8BED\u8A00',
  sys_prompt_h:'\u7CFB\u7EDF\u63D0\u793A\u8BCD',temperature:'\u6E29\u5EA6',
  max_tokens:'\u6700\u5927\u4EE4\u724C\u6570',top_p:'Top-P',
  api_key_label:'API \u5BC6\u9470\uFF08Bearer \u4EE4\u724C\uFF09',
  api_key_ph:'\u672A\u8BBE\u7F6E\u5219\u7559\u7A7A',
  server_url:'\u670D\u52A1\u5668\u5730\u5740',
  clear_chat:'\uD83D\uDDD1 \u6E05\u9664\u5BF9\u8BDD',
  agent_tools_h:'\u4EE3\u7406 / \u5DE5\u5177',enable_tool:'\u542F\u7528\u5DE5\u5177\u8C03\u7528',
  tool_demo:'\u5185\u7F6E\u6F14\u793A\u5DE5\u5177\uFF1A\u8BA1\u7B97\u5668\u3001\u83B7\u53D6\u65E5\u671F\u65F6\u95F4',
  greeting:'\u4F60\u597D\uFF01\u6211\u7531 <strong>llm.cpp</strong> \u9A71\u52A8\uFF0C\u6709\u4EC0\u4E48\u53EF\u4EE5\u5E2E\u52A9\u4F60\u7684\uFF1F',
  msg_ph:'\u8F93\u5165\u6D88\u606F\u2026\uFF08Enter \u53D1\u9001\uFF0CShift+Enter \u6362\u884C\uFF09',
  chat_cleared:'\u5BF9\u8BDD\u5DF2\u6E05\u9664\u3002',
  attach_img:'\u9644\u52A0\u56FE\u7247',attach_file:'\u9644\u52A0\u6587\u4EF6',
  tool_calls_h:'\uD83D\uDD27 \u5DE5\u5177\u8C03\u7528 \u2014 \u6267\u884C\u524D\u8BF7\u786E\u8BA4',
  allow:'\u2713 \u5141\u8BB8',deny:'\u2717 \u62D2\u7EDD',denied:'\u5DF2\u62D2\u7EDD',
  tool_not_impl:'\u6F14\u793A\u4E2D\u672A\u5B9E\u73B0\u6B64\u5DE5\u5177\u3002',
  file_prefix:'\u6587\u4EF6\uFF1A',img_suffix:' \u5F20\u56FE\u7247'
},
ja:{
  title:'llm.cpp \u30C1\u30E3\u30C3\u30C8',
  subtitle:'\u8EFD\u91CF LLM \u63A8\u8AD6\u30A8\u30F3\u30B8\u30F3',
  language:'\u8A00\u8A9E',
  sys_prompt_h:'\u30B7\u30B9\u30C6\u30E0\u30D7\u30ED\u30F3\u30D7\u30C8',
  temperature:'\u6E29\u5EA6',max_tokens:'\u6700\u5927\u30C8\u30FC\u30AF\u30F3\u6570',
  top_p:'Top-P',
  api_key_label:'API\u30AD\u30FC\uFF08Bearer\u30C8\u30FC\u30AF\u30F3\uFF09',
  api_key_ph:'\u672A\u8A2D\u5B9A\u306E\u5834\u5408\u306F\u7A7A\u767D',
  server_url:'\u30B5\u30FC\u30D0\u30FCURL',
  clear_chat:'\uD83D\uDDD1 \u30C1\u30E3\u30C3\u30C8\u3092\u30AF\u30EA\u30A2',
  agent_tools_h:'\u30A8\u30FC\u30B8\u30A7\u30F3\u30C8 / \u30C4\u30FC\u30EB',
  enable_tool:'\u30C4\u30FC\u30EB\u547C\u3073\u51FA\u3057\u3092\u6709\u52B9\u5316',
  tool_demo:'\u5185\u8535\u30C7\u30E2\u30C4\u30FC\u30EB\uFF1A\u8A08\u7B97\u6A5F\u3001\u65E5\u6642\u53D6\u5F97',
  greeting:'\u3053\u3093\u306B\u3061\u306F\uFF01<strong>llm.cpp</strong> \u3067\u52D5\u3044\u3066\u3044\u307E\u3059\u3002\u4F55\u304B\u304A\u624B\u4F1D\u3044\u3067\u304D\u307E\u3059\u304B\uFF1F',
  msg_ph:'\u30E1\u30C3\u30BB\u30FC\u30B8\u2026\uFF08Enter\u3067\u9001\u4FE1\u3001Shift+Enter\u3067\u6539\u884C\uFF09',
  chat_cleared:'\u30C1\u30E3\u30C3\u30C8\u3092\u30AF\u30EA\u30A2\u3057\u307E\u3057\u305F\u3002',
  attach_img:'\u753B\u50CF\u3092\u6DFB\u4ED8',attach_file:'\u30D5\u30A1\u30A4\u30EB\u3092\u6DFB\u4ED8',
  tool_calls_h:'\uD83D\uDD27 \u30C4\u30FC\u30EB\u547C\u3073\u51FA\u3057 \u2014 \u5B9F\u884C\u524D\u306B\u78BA\u8A8D',
  allow:'\u2713 \u8A31\u53EF',deny:'\u2717 \u62D2\u5426',denied:'\u62D2\u5426\u3055\u308C\u307E\u3057\u305F',
  tool_not_impl:'\u3053\u306E\u30C4\u30FC\u30EB\u306F\u30C7\u30E2\u3067\u306F\u672A\u5B9F\u88C5\u3067\u3059\u3002',
  file_prefix:'\u30D5\u30A1\u30A4\u30EB\uFF1A',img_suffix:' \u679A\u306E\u753B\u50CF'
}
};
var curLang=localStorage.getItem('llmcpp_lang')||'en';
function tr(key){var d=LANG[curLang]||LANG.en;return Object.prototype.hasOwnProperty.call(d,key)?d[key]:(Object.prototype.hasOwnProperty.call(LANG.en,key)?LANG.en[key]:key);}
function applyLang(){
  document.getElementById('html-root').lang=curLang;
  document.getElementById('page-title').textContent=tr('title');
  document.querySelectorAll('[data-i18n]').forEach(function(el){el.textContent=tr(el.getAttribute('data-i18n'));});
  document.querySelectorAll('[data-i18n-ph]').forEach(function(el){el.placeholder=tr(el.getAttribute('data-i18n-ph'));});
  document.querySelectorAll('[data-i18n-title]').forEach(function(el){el.title=tr(el.getAttribute('data-i18n-title'));});
  var gb=document.getElementById('greeting-bubble');if(gb)gb.innerHTML=tr('greeting');
  var sel=document.getElementById('lang');if(sel)sel.value=curLang;
}
function setLang(lang){curLang=lang;localStorage.setItem('llmcpp_lang',lang);applyLang();}
// ---- App state ----
var BASE=(window.location.protocol==='http:'||window.location.protocol==='https:')
  ?window.location.origin:'http://localhost:8080';
var srvEl=document.getElementById('srvurl');
srvEl.value=BASE;srvEl.placeholder='http://localhost:8080';
var history=[];
var imgs=[];
var attachedFiles=[];
var busy=false;
var _toolCallData=[];
// ---- Utilities ----
function autoH(el){el.style.height='auto';el.style.height=Math.min(el.scrollHeight,140)+'px';}
function esc(s){return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');}
function fmt(s){
  s=s.replace(/```(\w*)\n([\s\S]*?)```/g,function(_,l,c){return'<pre><code>'+esc(c.trim())+'</code></pre>';});
  s=s.replace(/`([^`\n]+)`/g,'<code>$1</code>');
  s=s.replace(/\*\*(.+?)\*\*/g,'<strong>$1</strong>');
  s=s.replace(/\*(.+?)\*/g,'<em>$1</em>');
  s=s.replace(/\n/g,'<br>');
  return s;
}
function toast(msg,dur){
  var d=document.createElement('div');d.className='toast';d.textContent=msg;
  document.body.appendChild(d);setTimeout(function(){d.remove();},dur||3000);
}
function scrollBottom(){var m=document.getElementById('msgs');m.scrollTop=m.scrollHeight;}
function addMsg(role,htmlContent){
  var d=document.createElement('div');d.className='msg '+role;
  var av=document.createElement('div');av.className='avatar';av.textContent=role==='user'?'\uD83D\uDC64':'\uD83E\uDD16';
  var bubble=document.createElement('div');bubble.className='bubble';
  bubble.innerHTML=htmlContent;
  d.appendChild(av);d.appendChild(bubble);
  document.getElementById('msgs').appendChild(d);
  scrollBottom();
  return bubble;
}
function addThinking(){
  var d=document.createElement('div');d.className='msg assistant';
  var av=document.createElement('div');av.className='avatar';av.textContent='\uD83E\uDD16';
  var bub=document.createElement('div');bub.className='bubble';
  bub.innerHTML='<div class="dots"><span></span><span></span><span></span></div>';
  d.appendChild(av);d.appendChild(bub);
  document.getElementById('msgs').appendChild(d);
  scrollBottom();
  return d;
}
// ---- Image attachment ----
function attachImg(inp){
  var f=inp.files[0];if(!f)return;
  var r=new FileReader();
  r.onload=function(e){
    var dataUrl=e.target.result;
    imgs.push(dataUrl);
    var p=document.getElementById('imgprev');
    var idx=imgs.length-1;
    var th=document.createElement('div');th.className='img-thumb';
    var img=document.createElement('img');img.src=dataUrl;
    var btn=document.createElement('button');btn.className='rm';btn.textContent='\xD7';
    btn.addEventListener('click',function(){imgs[idx]=null;th.remove();});
    th.appendChild(img);th.appendChild(btn);
    p.appendChild(th);
  };
  r.readAsDataURL(f);inp.value='';
}
// ---- Text file attachment (max 512 KB) ----
var MAX_FILE_BYTES=512*1024;
function attachFile(inp){
  var f=inp.files[0];if(!f)return;
  if(f.size>MAX_FILE_BYTES){toast(f.name+' is too large (max 512 KB)');inp.value='';return;}
  var r=new FileReader();
  r.onload=function(e){
    var idx=attachedFiles.length;
    attachedFiles.push({name:f.name,content:e.target.result});
    var p=document.getElementById('fileprev');
    var chip=document.createElement('div');chip.className='file-chip';
    var span=document.createElement('span');span.textContent=f.name;
    var btn=document.createElement('button');btn.className='rm';btn.textContent='\xD7';
    btn.addEventListener('click',function(){attachedFiles[idx]=null;chip.remove();});
    chip.appendChild(span);chip.appendChild(btn);
    p.appendChild(chip);
  };
  r.readAsText(f,'UTF-8');inp.value='';
}
// ---- Chat actions ----
function clearChat(){
  history=[];
  var c=document.getElementById('msgs');c.innerHTML='';
  var d=document.createElement('div');d.className='msg assistant';
  var av=document.createElement('div');av.className='avatar';av.textContent='\uD83E\uDD16';
  var bub=document.createElement('div');bub.className='bubble';bub.textContent=tr('chat_cleared');
  d.appendChild(av);d.appendChild(bub);c.appendChild(d);
}
function getTools(){
  return[
    {type:'function',function:{name:'calculator',description:'Evaluate a mathematical expression.',
      parameters:{type:'object',properties:{expression:{type:'string',description:'Math expression (numbers and operators only)'}},required:['expression']}}},
    {type:'function',function:{name:'get_datetime',description:'Get the current date and time.',
      parameters:{type:'object',properties:{}}}}
  ];
}
// Safe math evaluator: only allows numbers, whitespace and +−*/^%() operators.
function safeEval(expr){
  if(!/^[0-9\s\+\-\*\/\.\(\)\^%]+$/.test(expr))throw new Error('Invalid expression');
  var safe=expr.replace(/\^/g,'**');
  return Function('"use strict";return('+safe+')')();
}
function execTool(tc,resultDiv){
  var fn=tc.function||{};var args={};
  try{args=JSON.parse(fn.arguments||'{}');}catch(e){}
  var res='';
  try{
    if(fn.name==='calculator'){res=String(safeEval(args.expression||'0'));}
    else if(fn.name==='get_datetime'){res=new Date().toLocaleString();}
    else{res=tr('tool_not_impl');}
  }catch(e){res='Error: '+e.message;}
  resultDiv.innerHTML='';
  var strong=document.createElement('strong');strong.textContent='Result: ';
  var span=document.createElement('span');span.textContent=res;
  resultDiv.appendChild(strong);resultDiv.appendChild(span);
  history.push({role:'tool',tool_call_id:tc.id||'tc0',content:res});
  toast('Tool executed: '+fn.name);
}
function renderToolCalls(tcs,bubble){
  var block=document.createElement('div');block.className='tool-block';
  var h4=document.createElement('h4');h4.textContent=tr('tool_calls_h');
  block.appendChild(h4);
  tcs.forEach(function(tc){
    var startIdx=_toolCallData.length;
    _toolCallData.push(tc);
    var item=document.createElement('div');item.className='tool-item';
    var fn=tc.function||{};
    var fnDiv=document.createElement('div');fnDiv.className='fn';fnDiv.textContent=fn.name||'';
    var argsDiv=document.createElement('div');argsDiv.className='args';argsDiv.textContent=fn.arguments||'';
    var actions=document.createElement('div');actions.className='tool-actions';
    var res=document.createElement('div');res.style.cssText='font-size:.78rem;color:var(--dim);margin-top:4px';
    var allow=document.createElement('button');allow.className='btn btn-sm btn-ok';allow.textContent=tr('allow');
    allow.addEventListener('click',function(){execTool(_toolCallData[startIdx],res);});
    var deny=document.createElement('button');deny.className='btn btn-sm btn-danger';deny.textContent=tr('deny');
    deny.addEventListener('click',function(){res.textContent=tr('denied');});
    actions.appendChild(allow);actions.appendChild(deny);
    item.appendChild(fnDiv);item.appendChild(argsDiv);item.appendChild(actions);item.appendChild(res);
    block.appendChild(item);
  });
  bubble.appendChild(block);
}
async function send(){
  var text=document.getElementById('inp').value.trim();
  var attachedImgs=imgs.filter(Boolean);
  var activeFiles=attachedFiles.filter(Boolean);
  if(!text&&attachedImgs.length===0&&activeFiles.length===0)return;
  if(busy)return;
  document.getElementById('inp').value='';
  document.getElementById('inp').style.height='auto';
  imgs=[];document.getElementById('imgprev').innerHTML='';
  attachedFiles=[];document.getElementById('fileprev').innerHTML='';

  // Prepend file contents to the user message text
  var fullText=text;
  if(activeFiles.length>0){
    var prefix='';
    activeFiles.forEach(function(f){prefix+='['+tr('file_prefix')+f.name+']\n'+f.content+'\n\n';});
    fullText=prefix+text;
  }

  var content;
  if(attachedImgs.length>0){
    content=[];
    if(fullText)content.push({type:'text',text:fullText});
    attachedImgs.forEach(function(u){content.push({type:'image_url',image_url:{url:u}});});
  }else{content=fullText;}

  // Build display label shown in the chat bubble
  var parts=[];
  if(activeFiles.length>0)parts.push('['+activeFiles.map(function(f){return f.name;}).join(', ')+']');
  if(text)parts.push(text);
  if(attachedImgs.length>0)parts.push('['+attachedImgs.length+tr('img_suffix')+']');
  var displayText=parts.join(' ');

  history.push({role:'user',content:content});
  addMsg('user',esc(displayText));

  busy=true;document.getElementById('sendbtn').disabled=true;
  var th=addThinking();

  var sys=document.getElementById('sys').value.trim();
  var apiMsgs=[];
  if(sys)apiMsgs.push({role:'system',content:sys});
  apiMsgs=apiMsgs.concat(history);

  var payload={model:'llm.cpp',messages:apiMsgs,
    max_tokens:parseInt(document.getElementById('maxt').value)||512,
    temperature:parseFloat(document.getElementById('temp').value)||0.8,
    top_p:parseFloat(document.getElementById('topp').value)||0.9,
    stream:true};
  if(document.getElementById('agent').checked)payload.tools=getTools();

  var key=document.getElementById('apikey').value.trim();
  var hdrs={'Content-Type':'application/json'};
  if(key)hdrs['Authorization']='Bearer '+key;

  try{
    var r=await fetch((document.getElementById('srvurl').value||BASE)+'/v1/chat/completions',
      {method:'POST',headers:hdrs,body:JSON.stringify(payload)});
    th.remove();
    if(!r.ok){toast('HTTP '+r.status+': '+(await r.text()));busy=false;document.getElementById('sendbtn').disabled=false;return;}

    var aiText='';var toolCalls=[];
    var bubble=addMsg('assistant','');
    history.push({role:'assistant',content:''});

    var reader=r.body.getReader();var dec=new TextDecoder();var buf='';
    while(true){
      var chunk=await reader.read();if(chunk.done)break;
      buf+=dec.decode(chunk.value,{stream:true});
      var lines=buf.split('\n');buf=lines.pop();
      for(var i=0;i<lines.length;i++){
        var ln=lines[i];if(!ln.startsWith('data:'))continue;
        var d=ln.slice(5).trim();if(d==='[DONE]')continue;
        try{
          var j=JSON.parse(d);
          var delta=j.choices&&j.choices[0]&&j.choices[0].delta;
          if(delta&&delta.content){aiText+=delta.content;bubble.innerHTML=fmt(aiText);scrollBottom();}
          if(delta&&delta.tool_calls){
            if(Array.isArray(delta.tool_calls))toolCalls=delta.tool_calls;
          }
        }catch(e){}
      }
    }
    history[history.length-1].content=aiText;
    if(toolCalls.length>0)renderToolCalls(toolCalls,bubble);
  }catch(e){th.remove();toast('Error: '+e.message);}
  busy=false;document.getElementById('sendbtn').disabled=false;
}
// Apply language settings on page load
applyLang();
</script>
</body>
</html>
)HTML";
}

// ---- Server struct ----

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    std::string system_prompt = "You are a helpful assistant.";
    std::string api_key;        // if non-empty, require "Authorization: Bearer <api_key>"
    std::string model_name = "llm.cpp";  // ID returned by /v1/models
};

// Build a ChatML-format prompt from a messages array.
// Accepts an optional default system prompt used when none is provided.
static std::string build_chat_prompt(
    const std::vector<std::pair<std::string,std::string>>& messages,
    const std::string& default_system)
{
    std::string prompt;
    bool has_system = false;
    for (const auto& m : messages) {
        if (m.first == "system") { has_system = true; break; }
    }
    if (!has_system && !default_system.empty()) {
        prompt += "<|im_start|>system\n" + default_system + "<|im_end|>\n";
    }
    for (const auto& m : messages) {
        prompt += "<|im_start|>" + m.first + "\n" + m.second + "<|im_end|>\n";
    }
    prompt += "<|im_start|>assistant\n";
    return prompt;
}

// Run inference and accumulate output tokens into a string.
// model.kv_cache is cleared before generation and the model forward pass
// is run token-by-token without chat-template wrapping (caller provides the
// fully formatted prompt).
static std::string server_generate(Model& model, Sampler& sampler,
                                   const std::string& prompt,
                                   int max_tokens,
                                   int* out_prompt_tokens = nullptr,
                                   int* out_gen_tokens    = nullptr) {
    model.kv_cache.clear();

    std::vector<int> tokens = model.tokenizer.encode(prompt, false);
    if (out_prompt_tokens) *out_prompt_tokens = static_cast<int>(tokens.size());

    // Prefill
    for (int i = 0; i < static_cast<int>(tokens.size()); i++) {
        model.forward(tokens[i], i);
    }

    std::string output;
    int pos = static_cast<int>(tokens.size());
    std::vector<int> recent;
    const int recent_window = 64;

    for (int i = 0; i < max_tokens; i++) {
        float* logits = (i == 0) ? model.logits.data()
                                  : model.forward(recent.back(), pos - 1);
        int next = sampler.sample(logits, model.config.vocab_size, recent);
        if (model.tokenizer.is_eos_token(next)) break;
        recent.push_back(next);
        if (static_cast<int>(recent.size()) > recent_window)
            recent.erase(recent.begin());
        output += model.tokenizer.decode(next);
        pos++;
    }

    if (out_gen_tokens) *out_gen_tokens = static_cast<int>(recent.size());
    return output;
}

// Run inference with per-token streaming callback.
// callback(token_str) is called for each generated token.
static void server_generate_stream(
    Model& model, Sampler& sampler,
    const std::string& prompt, int max_tokens,
    std::function<bool(const std::string&)> callback)
{
    model.kv_cache.clear();
    std::vector<int> tokens = model.tokenizer.encode(prompt, false);
    for (int i = 0; i < static_cast<int>(tokens.size()); i++)
        model.forward(tokens[i], i);

    int pos = static_cast<int>(tokens.size());
    std::vector<int> recent;
    const int recent_window = 64;

    for (int i = 0; i < max_tokens; i++) {
        float* logits = (i == 0) ? model.logits.data()
                                  : model.forward(recent.back(), pos - 1);
        int next = sampler.sample(logits, model.config.vocab_size, recent);
        if (model.tokenizer.is_eos_token(next)) break;
        recent.push_back(next);
        if (static_cast<int>(recent.size()) > recent_window)
            recent.erase(recent.begin());
        std::string tok = model.tokenizer.decode(next);
        if (!callback(tok)) break;
        pos++;
    }
}

// Handle a single client connection.
static void handle_client(socket_t client_fd, Model& model, Sampler& sampler,
                           const ServerConfig& cfg) {
    std::string raw = http_recv_all(client_fd);
    if (raw.empty()) { CLOSE_SOCKET(client_fd); return; }

    // Parse request line
    size_t line_end = raw.find("\r\n");
    if (line_end == std::string::npos) { CLOSE_SOCKET(client_fd); return; }
    std::string req_line = raw.substr(0, line_end);

    // Extract method and path
    std::string method, path;
    {
        size_t sp1 = req_line.find(' ');
        if (sp1 == std::string::npos) { CLOSE_SOCKET(client_fd); return; }
        method = req_line.substr(0, sp1);
        size_t sp2 = req_line.find(' ', sp1 + 1);
        path = req_line.substr(sp1 + 1,
                               sp2 == std::string::npos ? std::string::npos
                                                        : sp2 - sp1 - 1);
    }

    // Handle CORS preflight
    if (method == "OPTIONS") {
        http_send(client_fd, "204 No Content", "text/plain", "");
        CLOSE_SOCKET(client_fd);
        return;
    }

    // Extract body (after \r\n\r\n)
    std::string body;
    size_t hdr_end = raw.find("\r\n\r\n");
    if (hdr_end != std::string::npos) body = raw.substr(hdr_end + 4);

    // ---- GET / or /index.html – serve HTML web chat interface ----
    if (method == "GET" && (path == "/" || path == "/index.html")) {
        std::string html = get_web_ui_html(cfg.port);
        http_send(client_fd, "200 OK", "text/html; charset=utf-8", html);
        CLOSE_SOCKET(client_fd);
        return;
    }

    // ---- API key authentication – applies only to /v1/* endpoints ----
    // The web UI (GET /) is always accessible so users can enter their key.
    bool is_api_path = (path.size() >= 4 && path.substr(0, 4) == "/v1/") ||
                       path == "/chat/completions";
    if (is_api_path && !auth_check(raw, cfg.api_key)) {
        http_send(client_fd, "401 Unauthorized", "application/json",
                  "{\"error\":{\"message\":\"Invalid or missing API key\","
                  "\"type\":\"invalid_request_error\",\"code\":\"invalid_api_key\"}}");
        CLOSE_SOCKET(client_fd);
        return;
    }

    // ---- GET /v1/models ----
    if (method == "GET" && path == "/v1/models") {
        long ts = static_cast<long>(time(nullptr));
        std::string id = json_escape(cfg.model_name);
        std::string resp =
            "{\"object\":\"list\",\"data\":[{\"id\":\"" + id + "\""
            ",\"object\":\"model\",\"created\":" + std::to_string(ts) +
            ",\"owned_by\":\"llm.cpp\"}]}";
        http_send(client_fd, "200 OK", "application/json", resp);
        CLOSE_SOCKET(client_fd);
        return;
    }

    // ---- POST /v1/chat/completions ----
    if (method == "POST" &&
        (path == "/v1/chat/completions" || path == "/chat/completions")) {

        // Use multimodal-aware message parser
        auto messages = json_parse_messages_multimodal(body);
        int max_tokens = static_cast<int>(json_get_num(body, "max_tokens", 512));
        if (max_tokens <= 0 || max_tokens > model.config.max_seq_len)
            max_tokens = 512;
        float temperature = static_cast<float>(
            json_get_num(body, "temperature", sampler.config.temperature));
        float top_p = static_cast<float>(
            json_get_num(body, "top_p", sampler.config.top_p));
        bool stream = json_get_bool(body, "stream", false);

        // Override sampler parameters per-request
        Sampler req_sampler = sampler;
        req_sampler.config.temperature = temperature;
        req_sampler.config.top_p       = top_p;

        // Build prompt from messages
        std::string prompt = build_chat_prompt(messages, cfg.system_prompt);

        long ts = static_cast<long>(time(nullptr));
        std::string model_id = json_escape(cfg.model_name);

        if (stream) {
            // Send chunked SSE response
            std::string stream_hdr =
                "HTTP/1.1 200 OK\r\n"
                "Content-Type: text/event-stream\r\n"
                "Cache-Control: no-cache\r\n"
                "Access-Control-Allow-Origin: *\r\n"
                "Access-Control-Allow-Headers: Content-Type, Authorization\r\n"
                "Connection: keep-alive\r\n"
                "\r\n";
#ifdef _WIN32
            send(client_fd, stream_hdr.c_str(),
                 static_cast<int>(stream_hdr.size()), 0);
#else
            send(client_fd, stream_hdr.c_str(), stream_hdr.size(), 0);
#endif
            std::string chunk_id = "chatcmpl-" + std::to_string(ts);

            // Role delta first
            {
                std::string role_chunk =
                    "{\"id\":\"" + chunk_id + "\","
                    "\"object\":\"chat.completion.chunk\","
                    "\"created\":" + std::to_string(ts) + ","
                    "\"model\":\"" + model_id + "\","
                    "\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\"},"
                    "\"finish_reason\":null}]}";
                sse_send(client_fd, role_chunk);
            }

            // Accumulate output to detect tool calls after generation
            std::string full_output;
            server_generate_stream(model, req_sampler, prompt, max_tokens,
                [&](const std::string& tok) -> bool {
                    full_output += tok;
                    std::string content_chunk =
                        "{\"id\":\"" + chunk_id + "\","
                        "\"object\":\"chat.completion.chunk\","
                        "\"created\":" + std::to_string(ts) + ","
                        "\"model\":\"" + model_id + "\","
                        "\"choices\":[{\"index\":0,\"delta\":{\"content\":\""
                        + json_escape(tok) + "\"},"
                        "\"finish_reason\":null}]}";
                    sse_send(client_fd, content_chunk);
                    return true;
                });

            // Check for tool call block in accumulated output
            std::string tool_json = extract_tool_call(full_output);
            if (!tool_json.empty()) {
                std::string tc_array = build_tool_calls_json(tool_json, ts);
                std::string tc_chunk =
                    "{\"id\":\"" + chunk_id + "\","
                    "\"object\":\"chat.completion.chunk\","
                    "\"created\":" + std::to_string(ts) + ","
                    "\"model\":\"" + model_id + "\","
                    "\"choices\":[{\"index\":0,"
                    "\"delta\":{\"tool_calls\":" + tc_array + "},"
                    "\"finish_reason\":\"tool_calls\"}]}";
                sse_send(client_fd, tc_chunk);
            }

            // Final stop chunk
            {
                std::string finish = tool_json.empty() ? "stop" : "tool_calls";
                std::string done_chunk =
                    "{\"id\":\"" + chunk_id + "\","
                    "\"object\":\"chat.completion.chunk\","
                    "\"created\":" + std::to_string(ts) + ","
                    "\"model\":\"" + model_id + "\","
                    "\"choices\":[{\"index\":0,\"delta\":{},"
                    "\"finish_reason\":\"" + finish + "\"}]}";
                sse_send(client_fd, done_chunk);
            }
            sse_send(client_fd, "[DONE]");

        } else {
            int prompt_tokens = 0, gen_tokens = 0;
            std::string content = server_generate(model, req_sampler, prompt,
                                                  max_tokens,
                                                  &prompt_tokens, &gen_tokens);

            // Check for tool call block
            std::string tool_json = extract_tool_call(content);
            std::string finish_reason = "stop";
            std::string choice_body;
            if (!tool_json.empty()) {
                finish_reason = "tool_calls";
                std::string tc_array = build_tool_calls_json(tool_json, ts);
                choice_body =
                    "{\"index\":0,"
                    "\"message\":{\"role\":\"assistant\","
                    "\"content\":" + (content.empty() ? "null" :
                                      "\"" + json_escape(content) + "\"") +
                    ",\"tool_calls\":" + tc_array + "},"
                    "\"finish_reason\":\"tool_calls\"}";
            } else {
                choice_body =
                    "{\"index\":0,"
                    "\"message\":{\"role\":\"assistant\","
                    "\"content\":\"" + json_escape(content) + "\"},"
                    "\"finish_reason\":\"stop\"}";
            }

            std::string resp =
                "{\"id\":\"chatcmpl-" + std::to_string(ts) + "\","
                "\"object\":\"chat.completion\","
                "\"created\":" + std::to_string(ts) + ","
                "\"model\":\"" + model_id + "\","
                "\"choices\":[" + choice_body + "],"
                "\"usage\":{\"prompt_tokens\":" + std::to_string(prompt_tokens) +
                ",\"completion_tokens\":" + std::to_string(gen_tokens) +
                ",\"total_tokens\":" + std::to_string(prompt_tokens + gen_tokens) +
                "}}";
            http_send(client_fd, "200 OK", "application/json", resp);
        }
        CLOSE_SOCKET(client_fd);
        return;
    }

    // ---- 404 ----
    http_send(client_fd, "404 Not Found", "application/json",
              "{\"error\":{\"message\":\"Not found\",\"type\":\"invalid_request_error\"}}");
    CLOSE_SOCKET(client_fd);
}

// Run the server loop. Blocks until killed.
static void run_server(Model& model, Sampler& sampler, const ServerConfig& cfg) {
#ifdef _WIN32
    WSADATA wsa;
    WSAStartup(MAKEWORD(2,2), &wsa);
#endif

    socket_t server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if (server_fd == SOCK_INVALID) {
        fprintf(stderr, "Server: failed to create socket\n");
        return;
    }

    int opt = 1;
#ifdef _WIN32
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR,
               reinterpret_cast<const char*>(&opt), sizeof(opt));
#else
    setsockopt(server_fd, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));
#endif

    struct sockaddr_in addr{};
    addr.sin_family      = AF_INET;
    addr.sin_port        = htons(static_cast<uint16_t>(cfg.port));
    addr.sin_addr.s_addr = INADDR_ANY;

    if (bind(server_fd, reinterpret_cast<struct sockaddr*>(&addr),
             sizeof(addr)) != 0) {
        fprintf(stderr, "Server: bind failed on port %d\n", cfg.port);
        CLOSE_SOCKET(server_fd);
        return;
    }
    if (listen(server_fd, 8) != 0) {
        fprintf(stderr, "Server: listen failed\n");
        CLOSE_SOCKET(server_fd);
        return;
    }

    fprintf(stderr, "Server listening on http://0.0.0.0:%d\n", cfg.port);
    fprintf(stderr, "Web chat UI:      http://localhost:%d/\n", cfg.port);
    fprintf(stderr, "OpenAI-compatible endpoints:\n");
    fprintf(stderr, "  GET  http://localhost:%d/v1/models\n", cfg.port);
    fprintf(stderr, "  POST http://localhost:%d/v1/chat/completions\n", cfg.port);
    if (!cfg.api_key.empty())
        fprintf(stderr, "API key:          enabled (set in --api-key)\n");
    fprintf(stderr, "Connect OpenWebUI via base URL: http://localhost:%d/v1\n\n",
            cfg.port);

    while (true) {
        socket_t client = accept(server_fd, nullptr, nullptr);
        if (client == SOCK_INVALID) continue;
        handle_client(client, model, sampler, cfg);
    }

    CLOSE_SOCKET(server_fd);
#ifdef _WIN32
    WSACleanup();
#endif
}

#endif // LLM_SERVER_H
