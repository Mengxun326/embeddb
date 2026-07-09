# Vexra — embedded vector database Docker image
# Build: docker build -t vexra .
# Run:   docker run -p 9020:9020 -v $(pwd)/data:/data vexra

FROM rust:1.80-slim-bookworm AS builder
WORKDIR /app
COPY . .
RUN cargo build --release -p vexra-cli && \
    cp target/release/vexra /usr/local/bin/vexra

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/bin/vexra /usr/local/bin/vexra
EXPOSE 9020
VOLUME /data
ENTRYPOINT ["vexra"]
CMD ["serve", "--host", "0.0.0.0", "--port", "9020"]
