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
#  include <arpa/inet.h>   // inet_pton, inet_ntop, inet_ntoa
#  include <sys/time.h>    // struct timeval (SO_RCVTIMEO timeout)
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
    <a id="dl-btn" class="btn btn-ghost btn-sm" style="width:100%;text-align:center;text-decoration:none;display:block" data-i18n="download_html">&#11015; Download UI</a>
    <div>
      <h3 data-i18n="agent_tools_h">Agent / Tools</h3>
      <label style="display:flex;align-items:center;gap:8px;cursor:pointer">
        <input type="checkbox" id="agent"> <span data-i18n="enable_tool">Enable tool calling</span>
      </label>
      <div style="font-size:.75rem;color:var(--dim);margin-top:6px" data-i18n="tool_demo">Built-in demo tools: calculator, get_datetime</div>
    </div>
    <div>
      <h3 data-i18n="upnp_h">UPnP Port Mapping</h3>
      <label style="display:flex;align-items:center;gap:8px;cursor:pointer">
        <input type="checkbox" id="upnp" onchange="toggleUpnp(this.checked)"> <span data-i18n="upnp_enable">Enable UPnP</span>
      </label>
      <div id="upnp-status" style="font-size:.75rem;color:var(--dim);margin-top:6px" data-i18n="upnp_desc">Map server port for external access via UPnP IGD</div>
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
  file_prefix:'File: ',img_suffix:' image(s)',
  download_html:'\u2B07 Download UI',
  upnp_h:'UPnP Port Mapping',upnp_enable:'Enable UPnP',
  upnp_desc:'Map server port for external access via UPnP IGD',
  upnp_on:'UPnP port mapping active',upnp_off:'UPnP port mapping disabled'
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
  file_prefix:'\u6587\u4EF6\uFF1A',img_suffix:' \u5F20\u56FE\u7247',
  download_html:'\u2B07 \u4E0B\u8F7D\u754C\u9762',
  upnp_h:'UPnP \u7AEF\u53E3\u6620\u5C04',upnp_enable:'\u542F\u7528 UPnP',
  upnp_desc:'\u901A\u8FC7 UPnP IGD \u6620\u5C04\u670D\u52A1\u5668\u7AEF\u53E3\u4EE5\u4FBF\u5916\u90E8\u8BBF\u95EE',
  upnp_on:'UPnP \u7AEF\u53E3\u6620\u5C04\u5DF2\u542F\u7528',upnp_off:'UPnP \u7AEF\u53E3\u6620\u5C04\u5DF2\u5173\u95ED'
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
  file_prefix:'\u30D5\u30A1\u30A4\u30EB\uFF1A',img_suffix:' \u679A\u306E\u753B\u50CF',
  download_html:'\u2B07 UI\u3092\u30C0\u30A6\u30F3\u30ED\u30FC\u30C9',
  upnp_h:'UPnP \u30DD\u30FC\u30C8\u30DE\u30C3\u30D4\u30F3\u30B0',upnp_enable:'UPnP \u3092\u6709\u52B9\u5316',
  upnp_desc:'UPnP IGD \u7D4C\u7531\u3067\u30B5\u30FC\u30D0\u30FC\u30DD\u30FC\u30C8\u3092\u30DE\u30C3\u30D4\u30F3\u30B0',
  upnp_on:'UPnP \u30DD\u30FC\u30C8\u30DE\u30C3\u30D4\u30F3\u30B0\u6709\u52B9',upnp_off:'UPnP \u30DD\u30FC\u30C8\u30DE\u30C3\u30D4\u30F3\u30B0\u7121\u52B9'
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
// Wire up the download button to /app.html on the server
(function(){var dl=document.getElementById('dl-btn');if(dl)dl.href=(BASE||'http://localhost:8080')+'/app.html';})();
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
// ---- UPnP toggle ----
async function toggleUpnp(enabled){
  var key=document.getElementById('apikey').value.trim();
  var hdrs={'Content-Type':'application/json'};
  if(key)hdrs['Authorization']='Bearer '+key;
  try{
    var r=await fetch((document.getElementById('srvurl').value||BASE)+'/v1/upnp',
      {method:'POST',headers:hdrs,body:JSON.stringify({enabled:enabled})});
    var j=await r.json();
    document.getElementById('upnp').checked=j.enabled;
    document.getElementById('upnp-status').textContent=j.enabled?tr('upnp_on'):tr('upnp_off');
    toast(j.enabled?tr('upnp_on'):tr('upnp_off'));
  }catch(e){toast('UPnP error: '+e.message);}
}
(async function(){
  try{
    var r=await fetch((document.getElementById('srvurl').value||BASE)+'/v1/upnp');
    var j=await r.json();
    document.getElementById('upnp').checked=j.enabled;
    if(j.enabled)document.getElementById('upnp-status').textContent=tr('upnp_on');
  }catch(e){}
})();
// Apply language settings on page load
applyLang();
</script>
</body>
</html>
)HTML";
}

// ---- UPnP IGD port mapping (SSDP + SOAP, no external dependencies) ----
//
// Discovers the Internet Gateway Device (router) on the local network via
// SSDP multicast, parses the description XML, and adds a TCP port mapping so
// that the llm.cpp server is reachable from the internet.

// --- XML helpers ---

// Return the text content of the first <tag>…</tag> occurrence.
static std::string xml_text(const std::string& xml, const std::string& tag) {
    std::string open = "<" + tag + ">";
    size_t s = xml.find(open);
    if (s == std::string::npos) {
        // Try <tag attr=…> form
        size_t ts = xml.find("<" + tag + " ");
        if (ts == std::string::npos) return "";
        size_t te = xml.find('>', ts);
        if (te == std::string::npos) return "";
        s = te + 1;
    } else {
        s += open.size();
    }
    size_t e = xml.find("</" + tag + ">", s);
    if (e == std::string::npos) return "";
    std::string v = xml.substr(s, e - s);
    size_t a = v.find_first_not_of(" \t\r\n");
    size_t b = v.find_last_not_of(" \t\r\n");
    return (a == std::string::npos) ? "" : v.substr(a, b - a + 1);
}

// Return the first <tag>…</tag> block that contains needle.
static std::string xml_block_with(const std::string& xml, const std::string& tag,
                                   const std::string& needle) {
    std::string ot = "<" + tag, ct = "</" + tag + ">";
    size_t pos = 0;
    while (pos < xml.size()) {
        size_t bs = xml.find(ot, pos);
        if (bs == std::string::npos) break;
        size_t te = xml.find('>', bs);
        if (te == std::string::npos) break;
        size_t be = xml.find(ct, te);
        if (be == std::string::npos) break;
        be += ct.size();
        if (xml.find(needle, te) < be) return xml.substr(bs, be - bs);
        pos = be;
    }
    return "";
}

// --- URL / socket helpers ---

static bool upnp_parse_url(const std::string& url,
                            std::string& host, int& port, std::string& path) {
    size_t sch = url.find("://");
    if (sch == std::string::npos) return false;
    std::string rest = url.substr(sch + 3);
    size_t pp = rest.find('/');
    std::string hp = (pp == std::string::npos) ? rest : rest.substr(0, pp);
    path = (pp == std::string::npos) ? "/" : rest.substr(pp);
    size_t cp = hp.rfind(':');
    if (cp == std::string::npos) { host = hp; port = 80; }
    else {
        host = hp.substr(0, cp);
        std::string ps = hp.substr(cp + 1);
        // Validate port string contains only digits
        for (char c : ps) { if (c < '0' || c > '9') { return false; } }
        port = ps.empty() ? 80 : atoi(ps.c_str());
    }
    return !host.empty() && port > 0;
}

// Apply receive/send timeouts (seconds) to a socket.
static void upnp_set_timeout(socket_t fd, int secs) {
#ifdef _WIN32
    DWORD ms = static_cast<DWORD>(secs * 1000);
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char*>(&ms), sizeof(ms));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char*>(&ms), sizeof(ms));
#else
    struct timeval tv; tv.tv_sec = secs; tv.tv_usec = 0;
    setsockopt(fd, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof(tv));
    setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof(tv));
#endif
}

// Connect a TCP socket to host:port (IPv4 numeric address).
static socket_t upnp_connect(const std::string& host, int port, int timeout_secs = 5) {
    socket_t fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd == SOCK_INVALID) return SOCK_INVALID;
    upnp_set_timeout(fd, timeout_secs);
    struct sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_port   = htons(static_cast<uint16_t>(port));
#ifdef _WIN32
    sa.sin_addr.s_addr = inet_addr(host.c_str());
    if (sa.sin_addr.s_addr == INADDR_NONE) { CLOSE_SOCKET(fd); return SOCK_INVALID; }
#else
    if (inet_pton(AF_INET, host.c_str(), &sa.sin_addr) <= 0) {
        CLOSE_SOCKET(fd); return SOCK_INVALID;
    }
#endif
    if (connect(fd, reinterpret_cast<struct sockaddr*>(&sa), sizeof(sa)) != 0) {
        CLOSE_SOCKET(fd); return SOCK_INVALID;
    }
    return fd;
}

// Send all bytes of buf via fd.
static void upnp_send_all(socket_t fd, const std::string& buf) {
    size_t sent = 0;
    while (sent < buf.size()) {
#ifdef _WIN32
        int n = send(fd, buf.c_str() + sent, static_cast<int>(buf.size() - sent), 0);
#else
        ssize_t n = send(fd, buf.c_str() + sent, buf.size() - sent, 0);
#endif
        if (n <= 0) break;
        sent += static_cast<size_t>(n);
    }
}

// Receive until connection closes; returns full response.
static std::string upnp_recv_all(socket_t fd) {
    std::string resp; char buf[4096];
    while (true) {
#ifdef _WIN32
        int n = recv(fd, buf, sizeof(buf), 0);
#else
        ssize_t n = recv(fd, buf, sizeof(buf), 0);
#endif
        if (n <= 0) break;
        resp.append(buf, static_cast<size_t>(n));
    }
    return resp;
}

// HTTP GET – returns response body on HTTP 200, "" otherwise.
static std::string upnp_http_get(const std::string& url) {
    std::string host; int port; std::string path;
    if (!upnp_parse_url(url, host, port, path)) return "";
    socket_t fd = upnp_connect(host, port, 5);
    if (fd == SOCK_INVALID) return "";
    std::string req = "GET " + path + " HTTP/1.0\r\nHost: " + host + ":" +
                      std::to_string(port) + "\r\nConnection: close\r\n\r\n";
    upnp_send_all(fd, req);
    std::string raw = upnp_recv_all(fd);
    CLOSE_SOCKET(fd);
    // Check HTTP status 200
    if (raw.size() < 12 || raw.substr(9, 3) != "200") return "";
    size_t he = raw.find("\r\n\r\n");
    return (he == std::string::npos) ? "" : raw.substr(he + 4);
}

// SOAP HTTP POST – returns (true, response_body) on HTTP 2xx.
static bool upnp_soap_call(const std::string& host, int port, const std::string& path,
                            const std::string& service_type,
                            const std::string& action_name,
                            const std::string& action_body,
                            std::string* out_body = nullptr) {
    std::string envelope =
        "<?xml version=\"1.0\"?>"
        "<s:Envelope xmlns:s=\"http://schemas.xmlsoap.org/soap/envelope/\""
        " s:encodingStyle=\"http://schemas.xmlsoap.org/soap/encoding/\">"
        "<s:Body><u:" + action_name + " xmlns:u=\"" + service_type + "\">"
        + action_body +
        "</u:" + action_name + "></s:Body></s:Envelope>";
    std::string soap_action = "\"" + service_type + "#" + action_name + "\"";
    std::string req =
        "POST " + path + " HTTP/1.0\r\n"
        "Host: " + host + ":" + std::to_string(port) + "\r\n"
        "Content-Type: text/xml; charset=\"utf-8\"\r\n"
        "SOAPAction: " + soap_action + "\r\n"
        "Content-Length: " + std::to_string(envelope.size()) + "\r\n"
        "Connection: close\r\n\r\n" + envelope;
    socket_t fd = upnp_connect(host, port, 10);
    if (fd == SOCK_INVALID) return false;
    upnp_send_all(fd, req);
    std::string raw = upnp_recv_all(fd);
    CLOSE_SOCKET(fd);
    if (raw.size() < 12) return false;
    // "HTTP/1.x NNN …" – validate the 3-digit status field
    bool digits_ok = true;
    for (size_t i = 9; i < 12 && i < raw.size(); i++)
        if (raw[i] < '0' || raw[i] > '9') { digits_ok = false; break; }
    int status = digits_ok ? atoi(raw.c_str() + 9) : 0;
    size_t he  = raw.find("\r\n\r\n");
    if (out_body && he != std::string::npos) *out_body = raw.substr(he + 4);
    return status >= 200 && status < 300;
}

// --- SSDP discovery ---

// Send SSDP M-SEARCH; returns the LOCATION URL of the first IGD found.
static std::string upnp_ssdp_discover(int timeout_secs = 3) {
#ifdef _WIN32
    socket_t fd = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
#else
    socket_t fd = socket(AF_INET, SOCK_DGRAM, 0);
#endif
    if (fd == SOCK_INVALID) return "";
    upnp_set_timeout(fd, timeout_secs);
    int one = 1;
#ifdef _WIN32
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char*>(&one), sizeof(one));
#else
    setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
#endif
    struct sockaddr_in local{};
    local.sin_family = AF_INET; local.sin_addr.s_addr = INADDR_ANY; local.sin_port = 0;
    bind(fd, reinterpret_cast<struct sockaddr*>(&local), sizeof(local));

    struct sockaddr_in dest{};
    dest.sin_family = AF_INET;
    dest.sin_port   = htons(1900);
#ifdef _WIN32
    dest.sin_addr.s_addr = inet_addr("239.255.255.250");
#else
    inet_pton(AF_INET, "239.255.255.250", &dest.sin_addr);
#endif
    const char* msearch =
        "M-SEARCH * HTTP/1.1\r\n"
        "HOST: 239.255.255.250:1900\r\n"
        "MAN: \"ssdp:discover\"\r\n"
        "MX: 3\r\n"
        "ST: urn:schemas-upnp-org:device:InternetGatewayDevice:1\r\n"
        "\r\n";
#ifdef _WIN32
    sendto(fd, msearch, static_cast<int>(strlen(msearch)), 0,
           reinterpret_cast<struct sockaddr*>(&dest), sizeof(dest));
#else
    sendto(fd, msearch, strlen(msearch), 0,
           reinterpret_cast<struct sockaddr*>(&dest), sizeof(dest));
#endif
    std::string location;
    char buf[2048] = {};
    for (int i = 0; i < 16 && location.empty(); i++) {
#ifdef _WIN32
        int n = recv(fd, buf, static_cast<int>(sizeof(buf) - 1), 0);
#else
        ssize_t n = recv(fd, buf, sizeof(buf) - 1, 0);
#endif
        if (n <= 0) break;
        buf[n] = '\0';
        // Case-insensitive scan for "location:" header (use str_to_lower helper)
        std::string resp(buf, static_cast<size_t>(n));
        std::string lo = str_to_lower(resp);
        size_t lp = lo.find("location:");
        if (lp != std::string::npos) {
            lp += 9;
            while (lp < resp.size() && (resp[lp]==' '||resp[lp]=='\t')) lp++;
            size_t le = resp.find("\r\n", lp);
            if (le == std::string::npos) le = resp.size();
            location = resp.substr(lp, le - lp);
            while (!location.empty() &&
                   (location.back()==' '||location.back()=='\r'||location.back()=='\n'))
                location.pop_back();
        }
    }
    CLOSE_SOCKET(fd);
    return location;
}

// --- IGD info ---

struct UpnpIgdInfo {
    std::string host;
    int         port = 0;
    std::string control_path;
    std::string service_type;
    bool        valid = false;
};

// Discover IGD and return control URL info.
static UpnpIgdInfo upnp_discover_igd(int timeout_secs = 3) {
    UpnpIgdInfo info;
    std::string location = upnp_ssdp_discover(timeout_secs);
    if (location.empty()) return info;

    std::string desc = upnp_http_get(location);
    if (desc.empty()) return info;

    // Try WANIPConnection and WANPPPConnection, both v1 and v2
    static const char* svc[] = {
        "WANIPConnection:1", "WANIPConnection:2",
        "WANPPPConnection:1", nullptr
    };
    std::string svc_block;
    for (int i = 0; svc[i]; i++) {
        svc_block = xml_block_with(desc, "service", svc[i]);
        if (!svc_block.empty()) {
            info.service_type = "urn:schemas-upnp-org:service:" + std::string(svc[i]);
            break;
        }
    }
    if (svc_block.empty()) return info;

    std::string ctrl = xml_text(svc_block, "controlURL");
    if (ctrl.empty()) return info;

    // Resolve ctrl relative to the base URL
    std::string bhost; int bport; std::string bpath;
    if (!upnp_parse_url(location, bhost, bport, bpath)) return info;
    if (ctrl[0] != '/') {
        size_t sl = bpath.rfind('/');
        ctrl = (sl == std::string::npos ? "/" : bpath.substr(0, sl + 1)) + ctrl;
    }
    info.host = bhost; info.port = bport; info.control_path = ctrl;
    info.valid = true;
    return info;
}

// Get the local IPv4 address used when connecting to the gateway.
static std::string upnp_local_ip(const std::string& gateway_host, int gateway_port) {
    socket_t fd = upnp_connect(gateway_host, gateway_port, 3);
    if (fd == SOCK_INVALID) return "";
    struct sockaddr_in local{};
#ifdef _WIN32
    int len = sizeof(local);
#else
    socklen_t len = sizeof(local);
#endif
    getsockname(fd, reinterpret_cast<struct sockaddr*>(&local), &len);
    CLOSE_SOCKET(fd);
#ifdef _WIN32
    const char* ip = inet_ntoa(local.sin_addr);
    return ip ? std::string(ip) : "";
#else
    char buf[64] = {};
    inet_ntop(AF_INET, &local.sin_addr, buf, sizeof(buf));
    return std::string(buf);
#endif
}

// Attempt UPnP IGD TCP port mapping.  Logs to stderr.  Returns true on success.
static bool upnp_map_port(int port) {
    fprintf(stderr, "UPnP: discovering IGD...\n");
    UpnpIgdInfo igd = upnp_discover_igd(3);
    if (!igd.valid) {
        fprintf(stderr, "UPnP: no IGD found (router may not support UPnP IGD)\n");
        return false;
    }
    fprintf(stderr, "UPnP: IGD at %s:%d%s  [%s]\n",
            igd.host.c_str(), igd.port, igd.control_path.c_str(),
            igd.service_type.c_str());

    std::string local_ip = upnp_local_ip(igd.host, igd.port);
    if (local_ip.empty() || local_ip == "0.0.0.0") {
        fprintf(stderr, "UPnP: could not determine local IP\n");
        return false;
    }
    fprintf(stderr, "UPnP: local IP: %s\n", local_ip.c_str());

    // GetExternalIPAddress (optional, for display only)
    std::string ext_body;
    upnp_soap_call(igd.host, igd.port, igd.control_path,
                   igd.service_type, "GetExternalIPAddress", "", &ext_body);
    std::string ext_ip = xml_text(ext_body, "NewExternalIPAddress");
    if (!ext_ip.empty()) fprintf(stderr, "UPnP: external IP: %s\n", ext_ip.c_str());

    // AddPortMapping
    std::string ps = std::to_string(port);
    std::string soap_body =
        "<NewRemoteHost></NewRemoteHost>"
        "<NewExternalPort>" + ps + "</NewExternalPort>"
        "<NewProtocol>TCP</NewProtocol>"
        "<NewInternalPort>" + ps + "</NewInternalPort>"
        "<NewInternalClient>" + local_ip + "</NewInternalClient>"
        "<NewEnabled>1</NewEnabled>"
        "<NewPortMappingDescription>llm.cpp</NewPortMappingDescription>"
        // Lease duration 0 = indefinite (router keeps mapping until reboot or explicit deletion)
        "<NewLeaseDuration>0</NewLeaseDuration>";
    std::string resp_body;
    bool ok = upnp_soap_call(igd.host, igd.port, igd.control_path,
                              igd.service_type, "AddPortMapping", soap_body, &resp_body);
    if (!ok || resp_body.find("UPnPError") != std::string::npos ||
               resp_body.find("Fault")     != std::string::npos) {
        std::string code = xml_text(resp_body, "errorCode");
        std::string desc = xml_text(resp_body, "errorDescription");
        fprintf(stderr, "UPnP: AddPortMapping failed%s%s\n",
                code.empty() ? "" : (" (code=" + code + " " + desc + ")").c_str(),
                !ok ? " (no HTTP 200)" : "");
        return false;
    }
    if (!ext_ip.empty())
        fprintf(stderr, "UPnP: mapped! External access: http://%s:%d/\n",
                ext_ip.c_str(), port);
    else
        fprintf(stderr, "UPnP: port %d mapped successfully\n", port);
    return true;
}

// Remove a previously added UPnP port mapping.  Returns true on success.
static bool upnp_unmap_port(int port) {
    fprintf(stderr, "UPnP: removing port mapping for port %d...\n", port);
    UpnpIgdInfo igd = upnp_discover_igd(3);
    if (!igd.valid) {
        fprintf(stderr, "UPnP: no IGD found\n");
        return false;
    }
    std::string ps = std::to_string(port);
    std::string soap_body =
        "<NewRemoteHost></NewRemoteHost>"
        "<NewExternalPort>" + ps + "</NewExternalPort>"
        "<NewProtocol>TCP</NewProtocol>";
    std::string resp_body;
    bool ok = upnp_soap_call(igd.host, igd.port, igd.control_path,
                              igd.service_type, "DeletePortMapping", soap_body, &resp_body);
    if (ok && resp_body.find("UPnPError") == std::string::npos &&
              resp_body.find("Fault")     == std::string::npos) {
        fprintf(stderr, "UPnP: port %d unmapped successfully\n", port);
        return true;
    }
    std::string code = xml_text(resp_body, "errorCode");
    std::string desc = xml_text(resp_body, "errorDescription");
    fprintf(stderr, "UPnP: DeletePortMapping failed%s%s\n",
            code.empty() ? "" : (" (code=" + code + " " + desc + ")").c_str(),
            !ok ? " (no HTTP 200)" : "");
    return false;
}

// ---- Server struct ----

// Global UPnP state – modified by the /v1/upnp endpoint and --upnp flag
static bool g_upnp_active = false;

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    std::string system_prompt = "You are a helpful assistant.";
    std::string api_key;        // if non-empty, require "Authorization: Bearer <api_key>"
    std::string model_name = "llm.cpp";  // ID returned by /v1/models
    bool upnp = false;          // attempt UPnP IGD port mapping on startup
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

    // ---- GET /app.html – download standalone single-file chat UI ----
    if (method == "GET" && path == "/app.html") {
        std::string html = get_web_ui_html(cfg.port);
        // Send with Content-Disposition so browsers save it as a file
        std::string resp_hdr =
            "HTTP/1.1 200 OK\r\n"
            "Content-Type: text/html; charset=utf-8\r\n"
            "Content-Disposition: attachment; filename=\"llm-chat.html\"\r\n"
            "Content-Length: " + std::to_string(html.size()) + "\r\n"
            "Access-Control-Allow-Origin: *\r\n"
            "Connection: close\r\n"
            "\r\n" + html;
        size_t sent = 0;
        while (sent < resp_hdr.size()) {
#ifdef _WIN32
            int n = send(client_fd, resp_hdr.c_str() + sent,
                         static_cast<int>(resp_hdr.size() - sent), 0);
#else
            ssize_t n = send(client_fd, resp_hdr.c_str() + sent,
                             resp_hdr.size() - sent, 0);
#endif
            if (n <= 0) break;
            sent += static_cast<size_t>(n);
        }
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

    // ---- GET /v1/upnp – UPnP port mapping status ----
    if (method == "GET" && path == "/v1/upnp") {
        std::string resp = "{\"enabled\":" +
                           std::string(g_upnp_active ? "true" : "false") + "}";
        http_send(client_fd, "200 OK", "application/json", resp);
        CLOSE_SOCKET(client_fd);
        return;
    }

    // ---- POST /v1/upnp – enable or disable UPnP port mapping ----
    if (method == "POST" && path == "/v1/upnp") {
        bool enable = json_get_bool(body, "enabled", false);
        if (enable && !g_upnp_active) {
            g_upnp_active = upnp_map_port(cfg.port);
        } else if (!enable && g_upnp_active) {
            upnp_unmap_port(cfg.port);
            g_upnp_active = false;
        }
        std::string resp = "{\"enabled\":" +
                           std::string(g_upnp_active ? "true" : "false") + "}";
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
    fprintf(stderr, "Standalone HTML:  http://localhost:%d/app.html  (save & open offline)\n",
            cfg.port);
    fprintf(stderr, "OpenAI-compatible endpoints:\n");
    fprintf(stderr, "  GET  http://localhost:%d/v1/models\n", cfg.port);
    fprintf(stderr, "  POST http://localhost:%d/v1/chat/completions\n", cfg.port);
    if (!cfg.api_key.empty())
        fprintf(stderr, "API key:          enabled (set in --api-key)\n");
    fprintf(stderr, "Connect OpenWebUI via base URL: http://localhost:%d/v1\n\n",
            cfg.port);

    // UPnP port mapping (non-blocking: runs before accept loop)
    if (cfg.upnp) {
        g_upnp_active = upnp_map_port(cfg.port);
        fprintf(stderr, "\n");
    }

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
