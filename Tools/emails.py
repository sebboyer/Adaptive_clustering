from email.header    import Header
from email.mime.text import MIMEText
from getpass         import getpass
from smtplib         import SMTP_SSL



def sendMe_email(email_text,passw):
	login, password = 'sebastienboyer54@gmail.com', passw
	recipients = [login]

	# create message
	msg = MIMEText(email_text, 'plain', 'utf-8')
	msg['Subject'] = Header('Python Update Email', 'utf-8')
	msg['From'] = login
	msg['To'] = ", ".join(recipients)

	# send it via gmail
	s = SMTP_SSL('smtp.gmail.com', 465, timeout=10)
	s.set_debuglevel(1)
	try:
	    s.login(login, password)
	    s.sendmail(msg['From'], recipients, msg.as_string())
	finally:
	    s.quit()