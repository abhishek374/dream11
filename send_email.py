import smtplib, ssl
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email import encoders
import pandas as pd
import csv
from datetime import date
def send_email_team(team1, team2, filepath,sender_email,receiver_email):
    smtp_server = "smtp.gmail.com"
    port = 587  # For starttls
    password = input("Type your password and press enter: ")
    # Read the attachment as pandas
    tabledf = pd.read_csv(filepath)
    tablehtml = tabledf.to_html()
    # Create the plain-text and HTML version of your message
    text =f"""\
    Hi,
    
    Please find attached the expected best team for Dream11 IPL match between {team1} and {team2}
    
    {tabledf}        
    
    Cheers!!!
    """

    html = """
    <html><body><p>Hi!</p>
    <p> Please find below the expected best team for Dream11 IPL match between {team1} and {team2}:</p>
    
    {table}
    
    <p>Cheers!,</p>
    <p>Abhishek</p>
    </body></html>
    """

    with open(filepath) as input_file:
        reader = csv.reader(input_file)
        data = list(reader)

    html = html.format(table=tablehtml, team1=team1, team2=team2)
    message = MIMEMultipart(
        "alternative", None, [ MIMEText(html, 'html')])
    #MIMEText(text),
    todays_date = date.today().strftime("%b-%d-%Y")
    message["Subject"] = f"Dream11 Team of the Day: {todays_date}"
    message["From"] = sender_email
    message["To"] = sender_email
    message["BCC"] = receiver_email

    filename = filepath  # In same directory as script

    # Open PDF file in binary mode
    with open(filename, "rb") as attachment:
        # Add file as application/octet-stream
        # Email client can usually download this automatically as attachment
        part = MIMEBase("application", "octet-stream")
        part.set_payload((attachment).read())

    # Encode file in ASCII characters to send by email
    encoders.encode_base64(part)

    # Add header as key/value pair to attachment part
    part.add_header('Content-Disposition', "attachment; filename= %s" % filename)

    # attach the instance 'p' to instance 'msg'
    message.attach(part)

    # Create a secure SSL context
    context = ssl.create_default_context()

    # Try to log in to server and send email
    try:
        server = smtplib.SMTP(smtp_server, port)
        server.ehlo()  # Can be omitted
        server.starttls(context=context)  # Secure the connection
        server.ehlo()  # Can be omitted
        server.login(sender_email, password)
        server.sendmail(
            sender_email, receiver_email, message.as_string()
        )
    except Exception as e:
        # Print any error messages to stdout
        print(e)
    finally:
        server.quit()

