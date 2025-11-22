from dotenv import load_dotenv
load_dotenv()

from nodes import *
from edges import *
from langchain_core.messages import convert_to_messages



# # try retriever
# retriever_tool.invoke({"query": "types of reward hacking"})

# # try generate_query_or_respond
# # input that does not require retriever
# input = {
#     "messages": [
#         {
#             "role": "user", "content": "hello!"
#         }
#     ]
# }
# generate_query_or_respond(input)["messages"][-1].pretty_print()

# # input that requires retriever
# input = {
#     "messages": [
#         {
#             "role": "user",
#             "content": "What does Lilian Weng say about types of reward hacking?",
#         }
#     ]
# }
# generate_query_or_respond(input)["messages"][-1].pretty_print()


# # try rewrite_question
input = {
    "messages": convert_to_messages(
        [
            {
                "role": "user",
                "content": "What does Lilian Weng say about types of reward hacking?",
            },
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "1",
                        "name": "retrieve_blog_posts",
                        "args": {"query": "types of reward hacking"},
                    }
                ],
            },
            {"role": "tool", "content": "meow", "tool_call_id": "1"},
        ]
    )
}

response = rewrite_question(input)
response["messages"][-1].pretty_print()
# todo - ideas on how to improve the prompt???

# # try generate_answer
# input = {
#     "messages": convert_to_messages(
#         [
#             {
#                 "role": "user",
#                 "content": "What does Lilian Weng say about types of reward hacking?",
#             },
#             {
#                 "role": "assistant",
#                 "content": "",
#                 "tool_calls": [
#                     {
#                         "id": "1",
#                         "name": "retrieve_blog_posts",
#                         "args": {"query": "types of reward hacking"},
#                     }
#                 ],
#             },
#             {
#                 "role": "tool",
#                 "content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
#                 "tool_call_id": "1",
#             },
#         ]
#     )
# }

# response = generate_answer(input)
# response["messages"][-1].pretty_print()


# try grade_documents conditional_edge
# irrelevant docs
# from langchain_core.messages import convert_to_messages

# input = {
#     "messages": convert_to_messages(
#         [
#             {
#                 "role": "user",
#                 "content": "What does Lilian Weng say about types of reward hacking?",
#             },
#             {
#                 "role": "assistant",
#                 "content": "",
#                 "tool_calls": [
#                     {
#                         "id": "1",
#                         "name": "retrieve_blog_posts",
#                         "args": {"query": "types of reward hacking"},
#                     }
#                 ],
#             },
#             {"role": "tool", "content": "meow", "tool_call_id": "1"},
#         ]
#     )
# }
# print(grade_documents(input))

# # relevant docs
# input = {
#     "messages": convert_to_messages(
#         [
#             {
#                 "role": "user",
#                 "content": "What does Lilian Weng say about types of reward hacking?",
#             },
#             {
#                 "role": "assistant",
#                 "content": "",
#                 "tool_calls": [
#                     {
#                         "id": "1",
#                         "name": "retrieve_blog_posts",
#                         "args": {"query": "types of reward hacking"},
#                     }
#                 ],
#             },
#             {
#                 "role": "tool",
#                 "content": "reward hacking can be categorized into two types: environment or goal misspecification, and reward tampering",
#                 "tool_call_id": "1",
#             },
#         ]
#     )
# }
# print(grade_documents(input))