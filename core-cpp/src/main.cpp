#include <iostream>
#include <thread>
#include <chrono>
#include <ctime>
#include <boost/asio.hpp>
#include <boost/beast/core.hpp>
#include <boost/beast/http.hpp>
#include <boost/beast/version.hpp>

namespace beast = boost::beast;
namespace http = beast::http;
namespace net = boost::asio;
using tcp = net::ip::tcp;

void handle_http_request(tcp::socket& socket) {
    try {
        beast::flat_buffer buffer;
        http::request<http::string_body> req;
        http::read(socket, buffer, req);
        
        http::response<http::string_body> res;
        res.set(http::field::server, "Stock Market Lab");
        res.set(http::field::access_control_allow_origin, "*");
        res.set(http::field::content_type, "application/json");
        
        std::string target = std::string(req.target());
        
        if (target == "/health") {
            res.result(http::status::ok);
            res.body() = "{\"status\":\"healthy\",\"timestamp\":\"" + std::to_string(std::time(nullptr)) + "\",\"services\":{\"backend\":true,\"database\":true,\"redis\":true}}";
        } else if (target == "/api/market/data") {
            res.result(http::status::ok);
            res.body() = "[{\"symbol\":\"AAPL\",\"price\":175.23,\"change\":2.5,\"changePercent\":1.45},{\"symbol\":\"GOOGL\",\"price\":142.50,\"change\":-1.2,\"changePercent\":-0.83},{\"symbol\":\"MSFT\",\"price\":380.75,\"change\":5.3,\"changePercent\":1.41}]";
        } else {
            res.result(http::status::not_found);
            res.body() = "{\"error\":\"Not found\"}";
        }
        
        res.prepare_payload();
        http::write(socket, res);
    } catch (...) {}
}

void http_server_thread() {
    try {
        net::io_context ioc{1};
        tcp::acceptor acceptor{ioc, {tcp::v4(), 8080}};
        
        std::cout << "Server started on port 8080" << std::endl;
        
        while (true) {
            tcp::socket socket{ioc};
            acceptor.accept(socket);
            std::thread([s = std::move(socket)]() mutable {
                handle_http_request(s);
            }).detach();
        }
    } catch (const std::exception& e) {
        std::cerr << "HTTP server error: " << e.what() << std::endl;
    }
}

int main() {
    try {
        http_server_thread();
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
    
    return 0;
}
