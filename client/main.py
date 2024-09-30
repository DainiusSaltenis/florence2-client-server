import requests

# Define the endpoint and parameters
url = "http://localhost:8080/detect-objects"
params = {"verbosity": "normal"}

# Open the image file in binary mode
file = open("/home/dainius/projects/florence2-client-server/resources/images/image-test2.jpg", "rb")

# Send the POST request with the image file
response = requests.post(url, files={"file": file})

file.close()

print(response)
print(response.text)
print(response.status_code)

# Print the JSON response from the server
print(response.json())
