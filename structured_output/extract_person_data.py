import pprint

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables from .env file
load_dotenv()


# llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash")
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

json_schema = {
    "title": "person",
    "description": "person profile",
    "type": "object",
    "properties": {
        "first_name": {
            "type": "string",
            "description": "The first name of the person",
            "default": ""
        },
        "last_name": {
            "type": "string",
            "description": "The last name of the person",
        },
        "birth_date": {
            "type": "string",
            "description": "the birth date of the person"
        },
        "companies": {
            "type": "array",
            "description": "a list of companies the person associated with",
            "items": {
                "type": "object",
                "properties": {
                    "company_name": {
                        "type": "string",
                        "description": "the company the person works at"
                    },
                    "role": {
                        "type": "string",
                        "description": "the role of the person in the company"
                    },
                    "industry": {
                        "type": "string",
                        "description": "the industry of the company",
                        "enum": ["automotive", "healthcare", "retail", "software", "governmental", "hardware"]
                    }
                }
            }
        },
        "education": {
            "type": "array",
            "description": "a list of universities the person graduated",
            "items": {
                "type": "object",
                "properties": {
                    "university_name": {
                        "type": "string",
                        "description": "the name of the university"
                    },
                    "degree": {
                        "type": "string",
                        "description": "the degree"
                    }
                }
            }
        }
    },
    "required": ["last_name", "first_name"],
}
# structured_llm = llm.with_structured_output(json_schema)

text = """
Elon Reeve Musk (born June 28, 1971) is a businessman known 
for his key roles in the space company SpaceX and the 
automotive company Tesla, Inc. 
His other involvements include ownership of X Corp., 
the company that operates the social media platform X (formerly Twitter), 
and his role in the founding of the Boring Company, xAI, Neuralink, and OpenAI. 
Musk is the wealthiest individual in the world; 
as of December 2024, Forbes estimates his net worth to be US$432 billion.
"""

text1 = """
Donald John Trump (born June 14, 1946) is an American politician, media personality, and businessman who is the 47th president of the United States. A member of the Republican Party, he served as the 45th president from 2017 to 2021.

Born into a wealthy family in New York City, Trump graduated from the University of Pennsylvania in 1968 with a bachelor's degree in economics. He became the president of his family's real estate business in 1971, renamed it the Trump Organization, and began acquiring and building skyscrapers, hotels, casinos, and golf courses. He launched side ventures, many licensing the Trump name, and filed for six business bankruptcies in the 1990s and 2000s. From 2004 to 2015, he hosted the reality television show The Apprentice, bolstering his fame as a billionaire. Presenting himself as a political outsider, Trump won the 2016 presidential election against Democratic Party nominee Hillary Clinton.

During his first presidency, Trump imposed a travel ban on seven Muslim-majority countries, expanded the Mexico–United States border wall, and enforced a family separation policy on the border. He rolled back environmental and business regulations, signed the Tax Cuts and Jobs Act, and appointed three Supreme Court justices. In foreign policy, Trump withdrew the U.S. from agreements on climate, trade, and Iran's nuclear program, and initiated a trade war with China. In response to the COVID-19 pandemic from 2020, he downplayed its severity, contradicted health officials, and signed the CARES Act. After losing the 2020 presidential election to Joe Biden, Trump attempted to overturn the result, culminating in the January 6 Capitol attack in 2021. He was impeached in 2019 for abuse of power and obstruction of Congress, and in 2021 for incitement of insurrection; the Senate acquitted him both times.

In 2023, Trump was found liable in civil cases for sexual abuse and defamation and for business fraud. He was found guilty of falsifying business records in 2024, making him the first U.S. president convicted of a felony. After winning the 2024 presidential election against Kamala Harris, he was sentenced to a penalty-free discharge, and two felony indictments against him for retention of classified documents and obstruction of the 2020 election were dismissed without prejudice. A racketeering case related to the 2020 election in Georgia is pending.

Trump began his second presidency by initiating mass layoffs of federal workers. He imposed tariffs on nearly all countries at the highest level since the Great Depression and signed the One Big Beautiful Bill Act. His administration's actions—including intimidation of political opponents and civil society, deportations of immigrants, and extensive use of executive orders—have drawn over 300 lawsuits challenging their legality. High-profile cases have underscored his broad interpretation of the unitary executive theory and have led to significant conflicts with the federal courts. In 2025, he authorized airstrikes on Iranian nuclear sites.

Since 2015, Trump's leadership style and political agenda—often referred to as Trumpism—have reshaped the Republican Party's identity. Many of his comments and actions have been characterized as racist or misogynistic, and he has made false or misleading statements and promoted conspiracy theories to an extent unprecedented in American politics. Trump's actions, especially in his second term, have been described as authoritarian and contributing to democratic backsliding. After his first term, scholars and historians ranked him as one of the worst presidents in American history.
"""
result = llm.with_structured_output(json_schema).invoke(text)
pprint.pprint(result)

# pip install streamlit
# import streamlit as st