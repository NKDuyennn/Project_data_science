import base64
import requests
from PIL import Image

def perform_ocr(image_path):
    response = requests.post(
        url="https://cca7-34-147-3-56.ngrok-free.app/ocr",
        json={
            "image_url": image_path,
        }
    )

    print("Response in = ", response.elapsed.total_seconds())
    if response.status_code == 200:
        return response.json().get("response_message")
    else:
        print("Error:", response.status_code, response.text)
        return None
    

image_path = "https://s3.cloud.cmctelecom.vn/nhattao1/2017/01/7719215_57525920a648bd5891a27741c926466f.jpg"
# show image
# im = Image.open(image_path)
# im.show()

result = perform_ocr(image_path)
# print(result)
if result:
    print("OCR Recognition Result:")
    print(result)
else:
    print("Error")