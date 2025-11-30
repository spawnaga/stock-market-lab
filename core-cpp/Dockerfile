FROM ubuntu:22.04

# Install dependencies
RUN apt-get update && apt-get install -y \
    g++ \
    cmake \
    pkg-config \
    libboost-all-dev \
    libwebsocketpp-dev \
    libssl-dev \
    libpq-dev \
    libpqxx-dev \
    libjsoncpp-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*



# Set working directory
WORKDIR /app

# Copy source files
COPY . .

# Build the application
RUN mkdir build && cd build && cmake .. && make

# Expose port
EXPOSE 8080

# Run the application
CMD ["./build/cpp_backend"]