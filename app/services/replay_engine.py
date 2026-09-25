import os
import time
import threading
from datetime import datetime

import pandas as pd


# ============================================================
# ATMOS — REAL-TIME AWS REPLAY ENGINE
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        ".."
    )
)


DATA_FILE = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "ml_features.csv"
)


class ReplayEngine:

    def __init__(self):

        self.data = None

        self.current_index = 0

        self.running = False

        self.thread = None

        self.interval = 2.0

        self.current_record = None

        self.total_records = 0

        self.started_at = None

        self.processed_records = 0

        self.lock = threading.Lock()


    # ========================================================
    # LOAD DATA
    # ========================================================

    def load_data(self):

        if self.data is None:

            self.data = pd.read_csv(
                DATA_FILE
            )

            self.data["timestamp"] = (
                pd.to_datetime(
                    self.data["timestamp"],
                    errors="coerce"
                )
            )

            self.data["station"] = (
                self.data["station"]
                .astype(str)
                .str.strip()
            )

            self.data = (
                self.data
                .sort_values(
                    [
                        "timestamp",
                        "station"
                    ]
                )
                .reset_index(
                    drop=True
                )
            )

            self.total_records = len(
                self.data
            )

        return self.data


    # ========================================================
    # GET CURRENT RECORD
    # ========================================================

    def get_current_record(self):

        with self.lock:

            if self.current_record is None:

                return None

            record = (
                self.current_record
                .copy()
            )

        return record


    # ========================================================
    # CONVERT RECORD
    # ========================================================

    def _convert_record(
        self,
        row
    ):

        record = {}

        for column in row.index:

            value = row[column]

            if pd.isna(value):

                value = None

            elif isinstance(
                value,
                pd.Timestamp
            ):

                value = (
                    value
                    .isoformat()
                )

            elif hasattr(
                value,
                "item"
            ):

                value = value.item()

            record[column] = value

        return record


    # ========================================================
    # PROCESS NEXT RECORD
    # ========================================================

    def process_next(self):

        self.load_data()

        with self.lock:

            if (
                self.current_index
                >= self.total_records
            ):

                self.current_index = 0


            row = self.data.iloc[
                self.current_index
            ]


            self.current_index += 1

            self.processed_records += 1


            self.current_record = (
                self._convert_record(
                    row
                )
            )


            return self.current_record


    # ========================================================
    # REPLAY LOOP
    # ========================================================

    def _replay_loop(self):

        while self.running:

            try:

                self.process_next()

                time.sleep(
                    self.interval
                )

            except Exception as error:

                print(
                    f"[REPLAY ERROR] {error}"
                )

                self.running = False


    # ========================================================
    # START
    # ========================================================

    def start(
        self,
        interval=2.0
    ):

        self.load_data()

        if self.running:

            return {
                "status": "already_running"
            }


        self.interval = max(
            float(interval),
            0.1
        )

        self.running = True

        self.started_at = (
            datetime.now()
            .isoformat()
        )

        self.thread = (
            threading.Thread(
                target=self._replay_loop,
                daemon=True
            )
        )

        self.thread.start()


        return {

            "status": "started",

            "interval_seconds":
                self.interval,

            "total_records":
                self.total_records,

            "started_at":
                self.started_at,

        }


    # ========================================================
    # STOP
    # ========================================================

    def stop(self):

        self.running = False

        return {
            "status": "stopped"
        }


    # ========================================================
    # RESET
    # ========================================================

    def reset(self):

        self.stop()

        with self.lock:

            self.current_index = 0

            self.current_record = None

            self.processed_records = 0

            self.started_at = None


        return {
            "status": "reset"
        }


    # ========================================================
    # STATUS
    # ========================================================

    def status(self):

        with self.lock:

            current_record = (
                self.current_record
            )

            return {

                "running":
                    self.running,

                "current_index":
                    self.current_index,

                "processed_records":
                    self.processed_records,

                "total_records":
                    self.total_records,

                "interval_seconds":
                    self.interval,

                "started_at":
                    self.started_at,

                "current_station":
                    (
                        current_record.get(
                            "station"
                        )
                        if current_record
                        else None
                    ),

                "current_timestamp":
                    (
                        current_record.get(
                            "timestamp"
                        )
                        if current_record
                        else None
                    ),

            }


# ============================================================
# GLOBAL REPLAY INSTANCE
# ============================================================

replay_engine = ReplayEngine()