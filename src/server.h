#ifndef LLM_SERVER_H
#define LLM_SERVER_H

// Minimal OpenAI-compatible HTTP server for llm.cpp.
// Implements GET /v1/models and POST /v1/chat/completions.
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

// Parse the "messages" array from a chat completion request.
// Returns a vector of (role, content) pairs.
static std::vector<std::pair<std::string,std::string>>
json_parse_messages(const std::string& json) {
    std::vector<std::pair<std::string,std::string>> msgs;
    size_t arr_start = json.find("\"messages\"");
    if (arr_start == std::string::npos) return msgs;
    size_t bracket = json.find('[', arr_start);
    if (bracket == std::string::npos) return msgs;
    // Walk the array, finding each object
    size_t pos = bracket + 1;
    int depth = 1;
    while (pos < json.size() && depth > 0) {
        if (json[pos] == '{') {
            // Find matching }
            size_t obj_start = pos;
            int obj_depth = 1;
            pos++;
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
            std::string role    = json_get_str(obj, "role");
            std::string content = json_get_str(obj, "content");
            if (!role.empty()) msgs.push_back({role, content});
        } else if (json[pos] == '[') {
            depth++;
            pos++;
        } else if (json[pos] == ']') {
            depth--;
            pos++;
        } else {
            pos++;
        }
    }
    return msgs;
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

// ---- Server struct ----

struct ServerConfig {
    std::string host = "0.0.0.0";
    int port = 8080;
    std::string system_prompt = "You are a helpful assistant.";
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

    // ---- GET /v1/models ----
    if (method == "GET" && path == "/v1/models") {
        long ts = static_cast<long>(time(nullptr));
        std::string resp =
            "{\"object\":\"list\",\"data\":[{\"id\":\"llm.cpp\""
            ",\"object\":\"model\",\"created\":" + std::to_string(ts) +
            ",\"owned_by\":\"llm.cpp\"}]}";
        http_send(client_fd, "200 OK", "application/json", resp);
        CLOSE_SOCKET(client_fd);
        return;
    }

    // ---- POST /v1/chat/completions ----
    if (method == "POST" &&
        (path == "/v1/chat/completions" || path == "/chat/completions")) {

        auto messages = json_parse_messages(body);
        int max_tokens = static_cast<int>(json_get_num(body, "max_tokens", 512));
        if (max_tokens <= 0 || max_tokens > model.config.max_seq_len)
            max_tokens = 512;
        float temperature = static_cast<float>(
            json_get_num(body, "temperature", sampler.config.temperature));
        bool stream = json_get_bool(body, "stream", false);

        // Override sampler temperature per-request
        Sampler req_sampler = sampler;
        req_sampler.config.temperature = temperature;

        // Build prompt from messages
        std::string prompt = build_chat_prompt(messages, cfg.system_prompt);

        long ts = static_cast<long>(time(nullptr));
        std::string model_id = "llm.cpp";

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

            server_generate_stream(model, req_sampler, prompt, max_tokens,
                [&](const std::string& tok) -> bool {
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

            // Final chunk
            {
                std::string done_chunk =
                    "{\"id\":\"" + chunk_id + "\","
                    "\"object\":\"chat.completion.chunk\","
                    "\"created\":" + std::to_string(ts) + ","
                    "\"model\":\"" + model_id + "\","
                    "\"choices\":[{\"index\":0,\"delta\":{},"
                    "\"finish_reason\":\"stop\"}]}";
                sse_send(client_fd, done_chunk);
            }
            sse_send(client_fd, "[DONE]");

        } else {
            int prompt_tokens = 0, gen_tokens = 0;
            std::string content = server_generate(model, req_sampler, prompt,
                                                  max_tokens,
                                                  &prompt_tokens, &gen_tokens);
            std::string resp =
                "{\"id\":\"chatcmpl-" + std::to_string(ts) + "\","
                "\"object\":\"chat.completion\","
                "\"created\":" + std::to_string(ts) + ","
                "\"model\":\"" + model_id + "\","
                "\"choices\":[{\"index\":0,"
                "\"message\":{\"role\":\"assistant\","
                "\"content\":\"" + json_escape(content) + "\"},"
                "\"finish_reason\":\"stop\"}],"
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
    fprintf(stderr, "OpenAI-compatible endpoints:\n");
    fprintf(stderr, "  GET  http://localhost:%d/v1/models\n", cfg.port);
    fprintf(stderr, "  POST http://localhost:%d/v1/chat/completions\n", cfg.port);
    fprintf(stderr, "Connect OpenWebUI via base URL: http://localhost:%d/v1\n\n", cfg.port);

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
