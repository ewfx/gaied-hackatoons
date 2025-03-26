import requests

# Define the URL of the FastAPI server
url = "http://127.0.0.1:8000/classify_email/"

headers = {"Content-Type": "application/json","Accept":"application/json"}

# Define the email text (input for classification)
email_text = "Congratulations! You've won a $1000 gift card. Click here to claim it."

# Define the data payload for the request (all fields should be passed as 'Form' data)
payload = {
    'request_type': 'Loan',
    'sub_request_type': 'loan application',
    'email_text': email_text,
    'msg': email_text,
    'request_data' : email_text


}

# Open the file to send with the request
files = {
    'attachment': open(r'L:\python projects\TestEmail\loan_adjustment.pdf', 'rb')  # Correct the file path
}

# Send a POST request to the server with the data and file
response = requests.post(url, data=payload , headers=headers)

# If the response is successful, print the classification result
if response.status_code == 200:
    result = response.json()  # Parse the JSON response
    print(f"Email Classification: {result['classification']}")
else:
    print(f"Failed to classify email. Status code: {response.status_code}, Error: {response.text}")
