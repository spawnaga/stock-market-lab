FROM rust:1.89-slim AS builder

WORKDIR /app

# Copy cargo files
COPY Cargo.toml Cargo.lock ./

# Copy source code
COPY src ./src

# Build release version
RUN cargo build --release

# Final stage - use bookworm for newer GLIBC
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/

# Copy the binary from builder stage
COPY --from=builder /app/target/release/order-book-simulator .

# Expose port
EXPOSE 8080

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:8080/health || exit 1

# Run the binary
CMD ["./order-book-simulator"]