from django.shortcuts import render,get_object_or_404
import json
from django.http import JsonResponse, StreamingHttpResponse
import time
from orders.models import Order
from .models import Conversation,Message
from .agents import run_support_agent
from .eventqueue import subscribe, unsubscribe, publish
from django.contrib.admin.views.decorators import staff_member_required

# Create your views here.

def chat(request,order_id):
    if request.method == 'POST':
        data = json.loads(request.body)
        user_message = data.get("message")

        print("USER MESSAGE: ",user_message)
        
    if not user_message:
        return JsonResponse({"error":"Empty msg "},status=400)
    
    order = get_object_or_404(Order,id = order_id,user=request.user)

    conversation, created = Conversation.objects.get_or_create(user=request.user,order=order)

    Message.objects.create(conversation=conversation,role='user',content=user_message)

    # Publish user message event to the Event Queue
    publish(conversation.id, {
        "type": "user_message",
        "message": user_message,
        "name": request.user.first_name if request.user.first_name else request.user.username
    })

    #send user message and conversation to LLM
    reply = run_support_agent(user_message,conversation.id,order.id,request.user.id)    
    #store the LLM reply
    Message.objects.create(conversation=conversation,role='assistant',content=reply)
    
    return JsonResponse({"reply":reply})


@staff_member_required
def dashboard(request):
    conversations = Conversation.objects.all().order_by('-created_at')
    print("CONVERSATIONS:==========> ",conversations)
    context = {
        "conversations":conversations
    }
    return render(request,'support/dashboard.html',context)

    
@staff_member_required
def conversation_detail(request,conversation_id):
    conversation = get_object_or_404(Conversation, id = conversation_id)
    messages = conversation.messages.order_by("created_at")
    agentlogs = conversation.agent_logs.order_by("created_at")
    # print("CONVERSATION===========================================>",conversation)
    # print("MESSAGES===========================================>",messages)
    # print("AGENTLOGS===========================================>",agentlogs)
    context ={
        "conversation":conversation,
        "messages":messages,
        "agentlogs":agentlogs
    }
    return render(request,'support/conversation_detail.html',context)

@staff_member_required
def conversation_stream(request,conversation_id):
    def event_stream(conversation_id):
        q = subscribe(conversation_id)

        try:
            while True:
                event = q.get() #wait for the next event

                yield f"data: {json.dumps(event)}\n\n"
        
        finally:
            unsubscribe(conversation_id,q)
            
    return StreamingHttpResponse(event_stream(conversation_id), content_type="text/event-stream")