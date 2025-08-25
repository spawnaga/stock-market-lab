#include <iostream>
#include <thread>
#include <chrono>
#include <set>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <json/json.h>
#include <pqxx/pqxx>

// Type definitions for WebSocket
typedef websocketpp::server<websocketpp::config::asio> server;

using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
using websocketpp::lib::bind;

// Global server instance
server ws_server;

// Track active connections
std::set<websocketpp::connection_hdl, std::owner_less<websocketpp::connection_hdl>> connections;

// Dummy market data generator
std::string generate_dummy_tick() {
    Json::Value root;
    root["symbol"] = "AAPL";
    root["price"] = 175.23 + (rand() % 100) / 100.0;
    root["volume"] = rand() % 1000000;
    root["timestamp"] = std::to_string(std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::high_resolution_clock::now().time_since_epoch()).count());
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, root);
}

// WebSocket connection handler
void on_open(server* s, websocketpp::connection_hdl hdl) {
    std::cout << "Client connected" << std::endl;
    connections.insert(hdl);
}

// WebSocket close handler
void on_close(server* s, websocketpp::connection_hdl hdl) {
    std::cout << "Client disconnected" << std::endl;
    connections.erase(hdl);
}

// WebSocket message handler
void on_message(server* s, websocketpp::connection_hdl hdl, server::message_ptr msg) {
    std::cout << "Received message: " << msg->get_payload() << std::endl;
    // Echo the message back
    s->send(hdl, msg->get_payload(), msg->get_opcode());
}

int main() {
    try {
        // Initialize the server
        ws_server.init_asio();
        
        // Set up handlers
        ws_server.set_open_handler(bind(&on_open, &ws_server, ::_1));
        ws_server.set_close_handler(bind(&on_close, &ws_server, ::_1));
        ws_server.set_message_handler(bind(&on_message, &ws_server, ::_1, ::_2));
        
        // Listen on port 8080
        ws_server.listen(8080);
        
        // Start the server
        ws_server.start_accept();
        
        std::cout << "Server started on port 8080" << std::endl;
        
        // Send dummy data periodically
        while (true) {
            std::string tick_data = generate_dummy_tick();
            
            // Broadcast to all connected clients
            for (auto& hdl : connections) {
                ws_server.send(hdl, tick_data, websocketpp::frame::opcode::text);
            }
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
    
    return 0;
}
