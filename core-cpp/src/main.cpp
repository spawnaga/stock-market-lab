#include <iostream>
#include <thread>
#include <chrono>
#include <websocketpp/config/asio_no_tls.hpp>
#include <websocketpp/server.hpp>
#include <json/json.h>
#include <pqxx/pqxx>
#include <curl/curl.h>
#include <sstream>
#include <iomanip>
#include <random>

// Type definitions for WebSocket
typedef websocketpp::server<websocketpp::config::asio> server;

using websocketpp::lib::placeholders::_1;
using websocketpp::lib::placeholders::_2;
using websocketpp::lib::bind;

// Global server instance
server ws_server;

// Global random generator for demo purposes
std::random_device rd;
std::mt19937 gen(rd());
std::uniform_real_distribution<> price_dist(150.0, 200.0);
std::uniform_int_distribution<> volume_dist(100000, 1000000);

// OHLCV data structure
struct OHLCVData {
    std::string symbol;
    double open;
    double high;
    double low;
    double close;
    long long volume;
    std::string timestamp;
};

// Callback function for CURL to write received data
static size_t WriteCallback(void* contents, size_t size, size_t nmemb, std::string* userp) {
    size_t totalSize = size * nmemb;
    userp->append((char*)contents, totalSize);
    return totalSize;
}

// Function to simulate fetching OHLCV data from an API
// In a real implementation, this would connect to Polygon.io, Alpha Vantage, etc.
OHLCVData fetch_ohlcv_data(const std::string& symbol) {
    OHLCVData data;
    data.symbol = symbol;
    
    // Generate realistic OHLCV data for demo purposes
    static double base_price = 175.0;
    static double prev_close = 175.0;
    
    // Random walk for price movement
    double change_percent = (gen() % 200 - 100) / 1000.0; // -0.1% to +0.1%
    double current_price = prev_close * (1.0 + change_percent);
    
    // Calculate OHLC values
    data.open = prev_close;
    data.close = current_price;
    
    // Determine high and low based on price movement
    if (current_price > prev_close) {
        data.high = std::max(current_price, prev_close + (prev_close * 0.005)); // Up to 0.5% higher
        data.low = std::min(prev_close, current_price - (current_price * 0.005)); // Down to 0.5% lower
    } else {
        data.high = std::max(prev_close, current_price + (current_price * 0.005)); // Up to 0.5% higher
        data.low = std::min(current_price, prev_close - (prev_close * 0.005)); // Down to 0.5% lower
    }
    
    // Ensure high >= max(open, close) and low <= min(open, close)
    data.high = std::max({data.high, data.open, data.close});
    data.low = std::min({data.low, data.open, data.close});
    
    // Generate realistic volume
    data.volume = volume_dist(gen);
    
    // Timestamp
    auto now = std::chrono::high_resolution_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
        now.time_since_epoch()).count();
    data.timestamp = std::to_string(ms);
    
    prev_close = current_price;
    
    return data;
}

// Convert OHLCV data to JSON
std::string ohlcv_to_json(const OHLCVData& data) {
    Json::Value root;
    root["symbol"] = data.symbol;
    root["open"] = data.open;
    root["high"] = data.high;
    root["low"] = data.low;
    root["close"] = data.close;
    root["volume"] = data.volume;
    root["timestamp"] = data.timestamp;
    
    Json::StreamWriterBuilder builder;
    return Json::writeString(builder, root);
}

// WebSocket connection handler
void on_open(server* s, websocketpp::connection_hdl hdl) {
    std::cout << "Client connected" << std::endl;
}

// WebSocket message handler
void on_message(server* s, websocketpp::connection_hdl hdl, server::message_ptr msg) {
    std::cout << "Received message: " << msg->get_payload() << std::endl;
    // Echo the message back
    s->send(hdl, msg->get_payload(), msg->get_opcode());
}

int main() {
    try {
        // Initialize curl
        curl_global_init(CURL_GLOBAL_DEFAULT);
        
        // Initialize the server
        ws_server.init_asio();
        
        // Set up handlers
        ws_server.set_open_handler(bind(&on_open, &ws_server, ::_1));
        ws_server.set_message_handler(bind(&on_message, &ws_server, ::_1, ::_2));
        
        // Listen on port 8080
        ws_server.listen(8080);
        
        // Start the server
        ws_server.start_accept();
        
        std::cout << "Server started on port 8080" << std::endl;
        
        // Send OHLCV data periodically
        std::vector<std::string> symbols = {"AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"};
        int symbol_index = 0;
        
        while (true) {
            // Fetch OHLCV data for current symbol
            OHLCVData data = fetch_ohlcv_data(symbols[symbol_index]);
            
            // Convert to JSON and broadcast
            std::string ohlcv_data = ohlcv_to_json(data);
            
            // Broadcast to all connected clients
            auto conns = ws_server.get_conns();
            for (auto it = conns.begin(); it != conns.end(); ++it) {
                ws_server.send(*it, ohlcv_data, websocketpp::frame::opcode::text);
            }
            
            // Cycle through symbols
            symbol_index = (symbol_index + 1) % symbols.size();
            
            std::this_thread::sleep_for(std::chrono::milliseconds(1000));
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error: " << e.what() << std::endl;
    }
    
    // Cleanup curl
    curl_global_cleanup();
    
    return 0;
}