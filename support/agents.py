from groq import Groq
from django.conf import settings
from .tools import get_order_details, get_refund_history, check_delivery_status
from .models import Conversation, Message, AgentLog
import json

# Initialize Groq Client
client = Groq(api_key=settings.API_KEY)
model = settings.MODEL

# SUPPORT SYSTEM PROMPT --> Maya's job description
SUPPORT_SYSTEM_PROMPT = """
You are Maya, a customer support agent at CoolBreeze AC.

You help customers with questions about their AC orders, deliveries, refunds, and related support issues.

The current order ID and current user ID will be provided to you in the system context.

Unless the customer explicitly mentions another order or another user, assume all order-related questions refer to the current order.

Your personality:
- Friendly and professional.
- Patient even when the customer is angry.
- Clear, concise, and helpful.
- Do not use emojis.

General behavior:
- Respond naturally to greetings, thanks, farewells, or casual conversation without using any tools.
- Examples include "hi", "hello", "good morning", "thanks", or "bye".
- Only use tools when you need factual information that is not already available.
- Do not call tools unnecessarily.
- If you need to use a tool, you may briefly acknowledge the customer's request before using it.

Tool usage rules:
- For questions about the current order, use the provided current order ID.
- Only ask the customer for an order ID if they clearly refer to a different order or if the current order cannot be determined.
- If the customer asks about order status, delivery, shipment, tracking, or order details, first call the get_order_details tool.
- If the customer asks about refunds or refund eligibility, first call get_refund_history.
- If delivery tracking information is needed, call check_delivery_status after obtaining the tracking number and carrier from get_order_details.
- If multiple tools are required, call them one at a time until you have enough information to answer.

Response rules:
- Never invent order details, tracking information, refund history, or delivery status.
- Base factual answers only on tool results.
- Never approve or deny a refund yourself. If a refund decision is requested, explain that the request will be reviewed by the appropriate team.
- Be empathetic and acknowledge the customer's concern before providing factual information.
- Once you have sufficient information from the tools, answer the customer's original question directly instead of asking for information you already know.

Your goal is to provide accurate, helpful, and trustworthy customer support while minimizing unnecessary tool usage.
"""

MANAGER_SYSTEM_PROMPT = """
You are a senior support manager at CoolBreeze AC.
A support agent has escalated a customer case to you for a refund decision.

Your responsibilities:
- Review the case summary carefully
- Consider the customer's refund history
- Make a fair and final refund decision
- Give a clear reason for your decision

Your decision options:
- Approve refund — if the case is genuine and within policy
- Deny refund — if the case is suspicious or outside policy
- Escalate to risk team — if you suspect fraud

Important rules:
- Be fair but firm
- Base decision on facts — not emotions
- Always give a specific reason for your decision
- Keep your response concise and professional
"""


# SUPPORT TOOLS --> Tool schemas that the AI agent will read
SUPPORT_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_order_details",
            "description": "Fetch complete order details including status, carrier, tracking number and days since order was placed. Use this when customer mentions their order or complains about delivery.",
            "parameters": {
                "type": "object",
                "properties": {
                    "order_id": {
                        "type": "integer",
                        "description": "The ID of the order to fetch details for"
                    }
                },
                "required": ["order_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_refund_history",
            "description": "Get complete refund history for a user. Use this before making any refund related decisions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "The user ID to check refund history for"
                    }
                },
                "required": ["user_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_delivery_status",
            "description": "Check current delivery status using tracking number and carrier. Use this when customer complains about delayed or missing delivery.",
            "parameters": {
                "type": "object",
                "properties": {
                    "tracking_number": {
                        "type": "string",
                        "description": "The shipment tracking number"
                    },
                    "carrier": {
                        "type": "string",
                        "description": "The carrier name for example BlueDart or Delhivery"
                    }
                },
                "required": ["tracking_number", "carrier"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_to_manager",
            "description": "Escalate the case to a senior manager for a refund decision. Use this when customer requests a refund. Always include order details, refund history and customer complaint in the case summary.",
            "parameters": {
                "type": "object",
                "properties": {
                    "case_summary": {
                        "type": "string",
                        "description": "Complete case summary including customer user_id, order details, refund history and the complaint reason."
                    }
                },
                "required": ["case_summary"]
            }
        }
    }
]


# execute_tool() --> bridge between llm and python functions
def execute_tool(tool_name, tool_input):
    if tool_name == "get_order_details":
        return get_order_details(tool_input["order_id"])

    if tool_name == "get_refund_history":
        return get_refund_history(tool_input["user_id"])

    if tool_name == "check_delivery_status":
        return check_delivery_status(
            tool_input["tracking_number"],
            tool_input["carrier"]
        )
    
    if tool_name == "escalate_to_manager":
        case_summary = tool_input["case_summary"]
        print("ESCALATING TO MANAGER",case_summary)
        decision=run_manager_agent(case_summary)
        print("MANAGER DECISION",decision)
        return decision

    return {"error": "tool not found"}


# Agent Loop --> While loop which loops until the task is done
def run_support_agent(user_message, conversation_id, order_id, user_id):
    conv = Conversation.objects.get(id=conversation_id)

    conversation_messages = []

    for msg in conv.messages.order_by("created_at"):
        conversation_messages.append({
            "role": msg.role,
            "content": msg.content
        })

    while True:
        # send this convo to llm
        context = f"\nCurrent order ID: {order_id}\nCurrent user ID: {user_id}"

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": SUPPORT_SYSTEM_PROMPT + context
                },
                *conversation_messages
            ],
            tools=SUPPORT_TOOLS,
            tool_choice="auto",
            max_tokens=1024,
        )
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        print("Stop reason -->", finish_reason)
        print("Content -->", message.content)

        if finish_reason == "stop":
            return message.content

        if finish_reason == "tool_calls":
            conversation_messages.append(message)

            for tool_call in message.tool_calls:

                tool_name = tool_call.function.name
                tool_input = json.loads(
                    tool_call.function.arguments
                )
                print("===================================")
                print("Tool called -->", tool_name)
                print("Tool input -->", tool_input)
                print("===================================")

                result = execute_tool(tool_name,tool_input)

                conversation_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
                })

                print("Tool result -->",result)
            continue

        else:
            return response.choices[0].message.content


def run_manager_agent(case_summary):
    manager_messages = [
        {"role":"user", "content":case_summary}
    ]

    while True:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": MANAGER_SYSTEM_PROMPT
                },
                *manager_messages
            ],
            # tools=SUPPORT_TOOLS,
            # tool_choice="auto",
            max_tokens=1024,
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        if finish_reason == "stop":
            return message.content

        if finish_reason == "tool_calls":
            manager_messages.append(message)
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_input = json.loads(tool_call.function.arguments)

                result = execute_tool(tool_name, tool_input)
                manager_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })
            continue