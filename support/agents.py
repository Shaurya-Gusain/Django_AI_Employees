from groq import Groq
from django.conf import settings
from .tools import get_order_details, get_refund_history, check_delivery_status, get_customer_risk_profile, search_knowledge_base
from .models import Conversation, Message, AgentLog
import json
from .eventqueue import publish, DONE

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
- When relaying a refund decision to the customer, communicate only the outcome and timeline. Do not share internal operational steps, monitoring flags, or audit notes.
- Never use bold text, bullet points, or any markdown formatting. Use plain text only.
- Keep replies concise and conversational (maximum 3-4 sentences, no long paragraphs).

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
- Keep your decision to: decision outcome + reason only.
- Do not include internal operational steps or next steps.
"""

RISK_SYSTEM_PROMPT = """
You are a fraud risk analyst at CoolBreeze AC.
A support manager has sent you a customer profile for risk assessment.

Your job:
- Analyse the customer's order and refund patterns
- Identify suspicious behaviour
- Return a clear risk verdict

Risk levels:
- LOW — genuine customer, normal behaviour
- MEDIUM — some suspicious signals, proceed with caution
- HIGH — clear fraud pattern, recommend denial

Your response format:
- Risk Level: LOW / MEDIUM / HIGH
- Key Signals: what you found suspicious or genuine
- Recommendation: what manager should do

Important:
- Be objective — base verdict on data only
- One bad refund does not make someone fraudulent
- Look for patterns — not isolated incidents
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
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search CoolBreeze AC company documents including refund policy, warranty policy, and product FAQs. Use this when customer asks about company policies, warranty coverage, warranty claims, refund eligibility, or any general product information that requires accurate company documentation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to find relevant information from company documents. Be specific — for example 'refund eligibility within 30 days' instead of just 'refund'."
                    }
                },
                "required": ["query"]
            }
        }
    }
]

RISK_TOOLS = [
    {
    "type": "function",
    "function": {
        "name": "get_customer_risk_profile",
        "description": "Get complete risk profile for a customer including order history, refund patterns and ratio. Use this to assess fraud risk for a customer.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_id": {
                    "type": "integer",
                    "description": "The user ID to assess risk for"
                }
            },
            "required": ["user_id"]
        }
    }
}
]

MANAGER_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "assess_fraud_risk",
            "description": "Consult the risk agent to assess fraud risk for a customer. Use this when refund request looks suspicious or customer has multiple refund requests.",
            "parameters": {
                "type": "object",
                "properties": {
                    "user_id": {
                        "type": "integer",
                        "description": "The user ID to assess fraud risk for"
                    }
                },
                "required": ["user_id"]
            }
        }
    }
]

# execute_tool() --> bridge between llm and python functions
# Pass conv through the chain to enable centralized logging for all agent events
def execute_tool(tool_name, tool_input, conv=None):
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
        print("=" * 50)
        print("ESCALATING TO MANAGER")
        print("=" * 50)
        case_summary = tool_input["case_summary"]
        print("ESCALATING TO MANAGER",case_summary)
        decision=run_manager_agent(case_summary, conv=conv)
        print("MANAGER DECISION",decision)
        return {"decision": decision}

    if tool_name == "assess_fraud_risk":
        print("=" * 50)
        print("CONSULTING RISK AGENT")
        print("=" * 50)
        user_id = tool_input["user_id"]
        verdict = run_risk_agent(user_id, conv=conv)
        return {"verdict": verdict}

    if tool_name == "get_customer_risk_profile":
        return get_customer_risk_profile(tool_input["user_id"])

    if tool_name == "search_knowledge_base":
        return search_knowledge_base(tool_input["query"])

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
            max_tokens=2048,
        )
        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        print("Stop reason -->", finish_reason)
        print("Content -->", message.content)

        if finish_reason == "stop":
            # Log the final support agent response before returning it
            AgentLog.objects.create(
                conversation=conv,
                event_type="final",
                message=message.content
            )
            publish(conv.id, {
                "type": "final",
                "message": message.content
            })
            publish(conv.id, DONE)
            return message.content

        if finish_reason == "tool_calls":
            conversation_messages.append(message)

            for tool_call in message.tool_calls:

                tool_name = tool_call.function.name
                tool_input = json.loads(
                    tool_call.function.arguments
                )
                AgentLog.objects.create(
                    conversation=conv,
                    event_type="tool_call",
                    message=f"Calling {tool_name} with {tool_input}"
                )
                publish(conv.id, {
                    "type": "tool_call",
                    "message": f"Calling {tool_name} with {tool_input}"
                })
                
                print("===================================")
                print("Tool called -->", tool_name)
                print("Tool input -->", tool_input)
                print("===================================")

                # Execute the tool and pass the conv object for potential nested agent logging
                result = execute_tool(tool_name, tool_input, conv=conv)

                # Log the tool result to the database
                AgentLog.objects.create(
                    conversation=conv,
                    event_type="tool_result",
                    message=f"{tool_name} returned: {str(result)[:200]}"
                )
                publish(conv.id, {
                    "type": "tool_result",
                    "message": f"{tool_name} returned: {str(result)[:200]}"
                })

                conversation_messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result)
                })

                print("Tool result -->",result)
            continue

    return ""  # safety fallback, should never reach here


def run_manager_agent(case_summary, conv=None):
    if conv:
        AgentLog.objects.create(
            conversation=conv,
            event_type="manager",
            message="Case received for review"
        )
        publish(conv.id, {
            "type": "manager",
            "message": "Case received for review"
        })

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
            tools=MANAGER_TOOLS,
            tool_choice="auto",
            max_tokens=1024,
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        print("MANAGER Stop reason -->", finish_reason)

        if finish_reason == "stop":
            # Log the manager's final decision
            if conv:
                AgentLog.objects.create(
                    conversation=conv,
                    event_type="manager",
                    message=message.content
                )
                publish(conv.id, {
                    "type": "manager",
                    "message": message.content
                })
            return message.content

        if finish_reason == "tool_calls":
            manager_messages.append(message)
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_input = json.loads(tool_call.function.arguments)

                print("MANAGER Tool called -->", tool_name)
                print("MANAGER Tool input -->", tool_input)

                # Log manager's tool call
                if conv:
                    if tool_name == "assess_fraud_risk":
                        msg = "Consulting risk agent for fraud assessment"
                    else:
                        msg = f"MANAGER Calling {tool_name} with {tool_input}"

                    AgentLog.objects.create(
                        conversation=conv,
                        event_type="manager",
                        message=msg
                    )
                    publish(conv.id, {
                        "type": "manager",
                        "message": msg
                    })

                result = execute_tool(tool_name, tool_input, conv=conv)
                
                # Log manager's tool result
                if conv:
                    AgentLog.objects.create(
                        conversation=conv,
                        event_type="tool_result",
                        message=f"MANAGER {tool_name} returned: {str(result)[:200]}"
                    )

                manager_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

                print("MANAGER Tool result -->", result)
            continue



def run_risk_agent(user_id, conv=None):
    if conv:
        AgentLog.objects.create(
            conversation=conv,
            event_type="risks",
            message=f"Starting fraud assessment for user ID {user_id}"
        )
        publish(conv.id, {
            "type": "risks",
            "message": f"Starting fraud assessment for user ID {user_id}"
        })

    risk_messages = [
        {
            "role": "user",
            "content": f"Please assess the fraud risk for user ID {user_id}. Use your tool to get their profile and return a verdict."
        }
    ]

    while True:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": RISK_SYSTEM_PROMPT
                },
                *risk_messages
            ],
            tools=RISK_TOOLS,
            tool_choice="auto",
            max_tokens=1024,
        )

        message = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        print("RISK Stop reason -->", finish_reason)

        if finish_reason == "stop":
            # Log the risk analyst's final verdict
            if conv:
                AgentLog.objects.create(
                    conversation=conv,
                    event_type="risks",
                    message=message.content
                )
                publish(conv.id, {
                    "type": "risks",
                    "message": message.content
                })
            return message.content

        if finish_reason == "tool_calls":
            risk_messages.append(message)

            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                tool_input = json.loads(tool_call.function.arguments)

                print("RISK Tool called -->", tool_name)
                print("RISK Tool input -->", tool_input)

                # Log risk agent's tool call
                if conv:
                    msg = f"Calling {tool_name} to get customer risk profile"
                    AgentLog.objects.create(
                        conversation=conv,
                        event_type="risks",
                        message=msg
                    )
                    publish(conv.id, {
                        "type": "risks",
                        "message": msg
                    })

                result = execute_tool(tool_name, tool_input, conv=conv)

                # Log risk agent's tool result
                if conv:
                    AgentLog.objects.create(
                        conversation=conv,
                        event_type="tool_result",
                        message=f"RISK {tool_name} returned: {str(result)[:200]}"
                    )

                risk_messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result)
                })

                print("RISK Tool result -->", result)
            continue