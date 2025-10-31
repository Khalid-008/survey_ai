# import requests
# from langchain.tools import tool
# from helper.log import Log

# logger = Log()


# @tool
# def create_workorder(
#     product_serial_number: str,
#     first_name_ar: str,
#     last_name_ar: str,
#     address: str,
#     national_id: str,
#     mobile_number: str,
#     email: str
# ):
#     """
#     create a new workorder based on customer input
#     """

#     validation = validate_create_order(
#         product_serial_number, address, national_id, mobile_number, email
#     )

#     if validation != "Success":
#         logger.error("validation failed: {}".format(validation))
#         return validation

#     logger.info("started creating workorder for {}".format(product_serial_number))

#     headers = {
#         "Content-Type": "application/json",
#         "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzYUBjaGFubmVscy5jb20uc2EiLCJSb2xlIjoiU3VwZXIgQWRtaW4iLCJGdWxsTmFtZUFyIjoi2KPYsdmI2Ykg2KfZhNi52YrYs9mJIiwiRnVsbE5hbWVFbiI6IkFyd2EgQeyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiJzYUBjaGFubmVscy5jb20uc2EiLCJSb2xlIjoiU3VwZXIgQWRtaW4iLCJGdWxsTmFtZUFyIjoi2KPYsdmI2Ykg2KfZhNi52YrYs9mJiwiRnVsbE5hbWVFbiI6IkFyd2EgQWxpc3NhIiwiaXNGaXJzdFRpbWUiOmZhbHNlLCJUeXBlIjoiVXNlciIsImlhdCI6MTY5NTg5MjAyMSwiZXhwIjoxNjk1ODkzODIxfQ.Lc6zLpI16RdDlPdHCC2zmKmpsqQ_TV2PFFV-dXJSzL4",
#     }
#     json_request = {
#         "productSerialNumber": product_serial_number,
#         "customer": {
#             "firstNameAr": first_name_ar,
#             "lastNameAr": last_name_ar,
#             "address": address,
#             "cityId": 1,
#             "nationalId": national_id,
#             "mobileNumber": mobile_number,
#             "email": email,
#             "preferredLanguage": "AR",
#         },
#         "productType": "OTHER",
#         "abuse": False,
#         "homeDelivery": False,
#         "ccDeliveryMethod": "DAL",
#         "dropInStore": 2,
#     }
#     logger.info("create workorder request: {}".format(json_request))
#     try:
#         r = requests.post(
#             "http://8.213.40.239:32652/api/sc/work-order/ai",
#             headers=headers,
#             json=json_request,
#             timeout=30,
#         )
#         r.raise_for_status()
#     except requests.exceptions.HTTPError as e:
#         logger.error("failed to create workorder: {}".format(e.response.text))
#         return "failed to create workorder: {}".format(
#             getattr(e.response, "text", str(e))
#         )
#     except requests.exceptions.RequestException as e:
#         logger.error("Error creating workorder: {}".format(str(e)))
#         return "Error creating workorder: {}".format(str(e))
#     except Exception as e:
#         logger.error("Error Exception creating workorder: {}".format(e))
#         return "Error Exception creating workorder: {}".format(e)
#     logger.info("created workorder")
#     return r.text


# def validate_create_order(
#     product_serial_number: str,
#     address: str,
#     national_id: str,
#     mobile_number: str,
#     email: str,
# ):
#     if product_serial_number is None or product_serial_number == "":
#         return {"ar": "رقم المنتج مطلوب", "en": "Serial number is required"}

#     if address is None or address == "":
#         return {"ar": "العنوان مطلوب", "en": "Address is required"}

#     if national_id is None or national_id == "":
#         return {"ar": "رقم الهوية مطلوب", "en": "National ID is required"}
#     if mobile_number is None or mobile_number == "":
#         return {"ar": "رقم الجوال مطلوب", "en": "Mobile number is required"}
#     if email is None or email == "":
#         return {"ar": "البريد الالكتروني مطلوب", "en": "Email is required"}

#     return "Success"
