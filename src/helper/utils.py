from flask import request,abort, jsonify
import xml.etree.ElementTree as ET
import re
import html
from langchain_community.document_loaders import TextLoader
from transformers import AutoTokenizer
import os
from langchain_core.messages import HumanMessage,AIMessage
from typing import Optional
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
from dotenv import load_dotenv
import json

load_dotenv()
model_name = os.getenv("MODEL")
tokenizer_model_name = os.getenv("TOKENIZER_MODEL")

def get_message_type(audio):
    message_type = ""
    if audio:
        message_type = "audio"
    else:
        message_type = "text"

    return message_type

def get_request_body():
    if not request.is_json:
        abort(400, "Request must be JSON")
    
    request_body = request.get_json()
    
    if 'request' not in request_body:
        abort(400, "Missing 'request' field in request body")
        
    req_data = request_body.get('request', {})
    if 'message' not in req_data:
        abort(400, "Missing required field 'message' in request['request']")
        
    return request_body

def get_response_body(response):
    return jsonify ({
        "response": response,
        "status": "success"
    })

def extract_content(content: str, key: Optional[str] = None):
    # Remove markdown code block markers
    xml_content = re.sub(r'```xml\n?|```\n?', '', content).strip()

    # Try extracting <response> tag content
    match = re.search(r"<response>.*?</response>", xml_content, flags=re.DOTALL)
    if match:
        response_only = match.group(0)
        parsed_data = parse_xml(response_only)
        if not parsed_data:
            return {"error": "Failed to parse XML content"}
        if key:
            value = parsed_data.get(key)
            if value is None:
                return {"error": f"Key '{key}' not found in XML content"}
            return html.unescape(value)
        # parsed_data is a dict, so return as-is
        return parsed_data

    # If no <response>, extract content after </think>
    match =  re.search(r"</think>\s*\n*\s*(.*?)['\"]\s+additional_kwargs=", content, flags=re.DOTALL)
    if match:
        return html.unescape(re.sub(r'^\\n\\n', '', match.group(1).strip()))

    return {"error": "No <response> tag or fallback content found"}

def parse_xml(xml_content):
    try:
        root = ET.fromstring(xml_content)
        
        # Extract all elements
        data = {}
        for child in root:
            data[child.tag] = child.text
        
        return data
    except ET.ParseError as e:
        print(f"XML parsing error: {e}")
        return None

def count_tokens(text):
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_model_name)
    tokens = tokenizer.encode(text)
    token_count = len(tokens)
    print(f"Token count: {token_count}\n")


def loadTextFile(path : str): 
    return TextLoader(path)

def pretty_print_message(message, indent=False):
    pretty_message = message.pretty_repr(html=True)
    if not indent:
        print(pretty_message)
        return
    indented = "\n".join("\t" + c for c in pretty_message.split("\n"))
    print(indented)


def remove_think_blocks(text):
    """
    Removes all <think>...</think> blocks from the input text.
    """
    return re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL).strip()

def strip_think_blocks(messages):
    cleaned_messages = []
    for msg in messages:
        # If it's an AIMessage, clean the content
        if isinstance(msg, AIMessage):
            if isinstance(msg.content, str):
                content = re.sub(r"<think>.*?</think>\s*", "", msg.content, flags=re.DOTALL).strip()
                cleaned_messages.append(AIMessage(content=content))
            else:
                cleaned_messages.append(msg)
        else:
            cleaned_messages.append(msg)
    return str(cleaned_messages)

def parse_langgraph_output(stream):
    results = []
    for key, value in stream.items():
        if key == "supervisor":
            continue
        messages = value.get("messages", [])
        for msg in messages:
            if isinstance(msg, str):
                results.append((key, msg))
            elif isinstance(msg, AIMessage):
                results.append((key, msg.content))
    return results

def select_tools(all_tools, tool_names, verbose=True):

    selected_tools = [tool for tool in all_tools if tool.name in tool_names]
    
    if verbose:
        if not selected_tools:
            print(f"None of the specified tools found: {tool_names}")
            print("Available tools:", [t.name for t in all_tools])
        else:
            missing_tools = [name for name in tool_names if name not in [t.name for t in selected_tools]]
            if missing_tools:
                print(f"Warning: These tools were not found: {missing_tools}")
            
            print(f"Selected tools ({len(selected_tools)}):")
            for tool in selected_tools:
                print(f"  - {tool.name}")
    
    return selected_tools

def detect_language(text):
    """
    Detect if text is English or Arabic
    
    Args:
        text (str): Input text to analyze
    
    Returns:
        str: "English", "Arabic", or "Unknown"
    """
    try:
        # Detect language code
        lang_code = detect(text)
        
        # Check if it's English or Arabic
        if lang_code == 'en':
            return 'English'
        elif lang_code == 'ar':
            return 'Arabic'
        else:
            return 'English'
            
    except LangDetectException:
        return "Unknown"


def extract_json(json_string: str):
    """
    Extract and parse JSON from various formats, handling escape sequences robustly.

    Args:
        json_string: String containing JSON (possibly with markdown blocks or escape issues)

    Returns:
        dict: Parsed JSON data

    Raises:
        ValueError: If input is empty/None or JSON cannot be parsed
    """
    # Handle None or empty input
    if not json_string or not isinstance(json_string, str):
        raise ValueError("Input must be a non-empty string")

    # Strip whitespace
    json_string = json_string.strip()

    # Handle empty string after stripping
    if not json_string:
        raise ValueError("Input string is empty after stripping whitespace")

    # Remove markdown code blocks (multiple formats)
    if json_string.startswith("```json"):
        json_string = json_string[len("```json"):].strip()
    elif json_string.startswith("```"):
        json_string = json_string[len("```"):].strip()

    if json_string.endswith("```"):
        json_string = json_string[:-3].strip()

    # Fix common escape sequence issues before parsing
    # Replace invalid escape sequences that cause "Invalid \escape" errors
    # This handles cases where LLM generates malformed JSON
    json_string = json_string.replace("\\'", "'")  # Invalid in JSON, replace with plain quote

    # Try to parse JSON
    try:
        data = json.loads(json_string)
    except (json.JSONDecodeError, ValueError) as e:
        # If parsing fails, try to extract JSON object from the string
        print(f"Initial JSON parsing failed: {e}")

        # Look for the first { and last }
        start_idx = json_string.find('{')
        end_idx = json_string.rfind('}')

        if start_idx == -1 or end_idx == -1 or start_idx >= end_idx:
            # No valid JSON object found
            raise ValueError(
                f"Could not find valid JSON object. Error: {str(e)}. "
                f"First 200 chars: {json_string[:200]}"
            )

        # Extract the JSON portion
        json_string = json_string[start_idx:end_idx + 1]

        # Fix escape sequences again on the extracted portion
        json_string = json_string.replace("\\'", "'")

        try:
            data = json.loads(json_string)
            print("Successfully parsed JSON after extraction")
        except (json.JSONDecodeError, ValueError) as e2:
            # Still failed, provide helpful error message
            raise ValueError(
                f"JSON parsing failed even after extraction. Error: {str(e2)}. "
                f"Extracted string (first 300 chars): {json_string[:300]}"
            )

    # Aggressively unescape HTML content in visualizations
    if 'visualizations' in data and isinstance(data['visualizations'], list):
        for viz in data['visualizations']:
            if isinstance(viz, dict) and 'chart_html' in viz and viz['chart_html']:
                html = viz['chart_html']

                # Keep unescaping until all escape sequences are gone
                # This handles double-escaped content from LLM responses
                max_iterations = 10  # Prevent infinite loops
                iteration = 0
                while iteration < max_iterations and (
                    '\\"' in html or '\\n' in html or '\\t' in html or '\\\\' in html
                ):
                    html = html.replace('\\"', '"')
                    html = html.replace('\\n', '\n')
                    html = html.replace('\\t', '\t')
                    html = html.replace('\\r', '\r')
                    html = html.replace('\\\\', '\\')
                    iteration += 1

                viz['chart_html'] = html

    return data