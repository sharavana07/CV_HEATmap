import websocket, json, time, signal, sys
import pandas as pd
from pathlib import Path

SYMBOL = "btcusdt"
URL = f"wss://stream.binance.com:9443/ws/{SYMBOL}@depth20@100ms" # 20 levels of bids/asks, 100ms update interval
OUT_DIR = Path("../data/raw_orderbook1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

buffer = []
FLUSH_EVERY = 500          # flush after 500 rows
FLUSH_INTERVAL = 30        # seconds – force flush even if buffer not full
last_flush_time = time.time()

# ------------------------------------------------------------
def on_message(ws, message):
    global last_flush_time
    
    #print("MESSAGE RECEIVED")


    msg = json.loads(message)
    ts = time.time()
    exchange_ts = msg.get("E")          # safe access
    local_ts = time.time_ns()
    
    # # ---------- DIAGNOSTIC PRINT ----------
    # # Print how many levels are actually in the message
    # print(len(msg.get("bids", [])), len(msg.get("asks", [])))
    # # ---------------------------------------

    bids = msg.get("bids", [])[:25] #25 levels of bids
    asks = msg.get("asks", [])[:25] #25 levels of asks  

    row = {
        "ts": ts,
        "exchange_ts": exchange_ts, # safe access
        "local_ts": local_ts, #local timestamp in nanoseconds
    }
    for i, (p, q) in enumerate(bids):
        row[f"bid_p_{i}"] = float(p)
        row[f"bid_q_{i}"] = float(q)
    for i, (p, q) in enumerate(asks):
        row[f"ask_p_{i}"] = float(p)
        row[f"ask_q_{i}"] = float(q)

    buffer.append(row)

    # Lightweight live log – overwrites the same line
    #print(f"\rMessages received: {len(buffer)}", end="", flush=True)

    # Flush if buffer full OR time interval passed
    now = time.time()
    if len(buffer) >= FLUSH_EVERY or (now - last_flush_time) >= FLUSH_INTERVAL:
        flush()
        last_flush_time = now

# ------------------------------------------------------------
def flush():
    global buffer
    if not buffer:
        return
    df = pd.DataFrame(buffer)
    # Unique filename: timestamp + millisecond counter
    fname = OUT_DIR / f"chunk_{int(time.time()*1000)}.parquet"
    df.to_parquet(fname)
    print(f"\n💾 Flushed {len(buffer)} rows → {fname}")
    buffer.clear()

# ------------------------------------------------------------
def on_open(ws):
    print(f"\n✅ Connected to Binance – streaming {SYMBOL}@depth20@100ms")

def on_error(ws, error):
    print("\n⚠️ WebSocket error:", repr(error))

def on_close(ws, close_status_code, close_msg):             
    print(f"\n🔌 WebSocket closed (code={close_status_code}). Flushing...")
    flush()

# ------------------------------------------------------------
def signal_handler(sig, frame):
    print("\n🛑 Interrupted by user. Flushing buffer and exiting...")
    flush()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# ------------------------------------------------------------
if __name__ == "__main__":
    RECONNECT_DELAY = 10  # seconds – don’t lower, prevents Binance rate-limit

    while True:
        try:
            ws = websocket.WebSocketApp(
                URL,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
            )
            # Reset timer
            last_flush_time = time.time()
            ws.run_forever()
        except KeyboardInterrupt:
            signal_handler(None, None)
        except Exception as e:
            print(f"\n💥 Connection lost: {e}")
            print(f"⏳ Reconnecting in {RECONNECT_DELAY}s...")
            flush()   # save whatever is left
            time.sleep(RECONNECT_DELAY)