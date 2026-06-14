from io import BytesIO
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app


def main():
    client = app.test_client()

    health = client.get("/health")
    assert health.status_code == 200, health.data.decode()
    assert health.get_json()["status"] == "healthy"

    home = client.get("/")
    assert home.status_code == 200, home.data.decode()


    with open("tests/before.jpg", "rb") as before_file, open("tests/after.jpg", "rb") as after_file:
        upload = client.post(
            "/api/upload",
            data={
                "before": (BytesIO(before_file.read()), "before.jpg"),
                "after": (BytesIO(after_file.read()), "after.jpg"),
            },
            content_type="multipart/form-data",
        )

    assert upload.status_code == 200, upload.data.decode()
    payload = upload.get_json()
    assert payload["success"] is True
    assert payload["analysis"]["regions_detected"] >= 0

    print("BuildWatch AI smoke test passed")


if __name__ == "__main__":
    main()
