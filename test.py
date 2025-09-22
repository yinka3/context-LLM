import unittest
from typing import Optional
from main.context import Context 
from shared.dtypes import MessageData, EntityData



class TestTier1ConversationFlow(unittest.TestCase):

    def setUp(self):
        """Set up a fresh context for each test."""
        self.context = Context()
        self.message_id_counter = 0

    def _process_message(self, msg: str):
        """Helper method to process a single message."""
        msg_data = MessageData(
            id=self.message_id_counter,
            role="user",
            message=msg,
            sentiment="unknown" # Sentiment is determined during processing
        )
        self.context.add(msg_data)
        self.message_id_counter += 1

    def test_student_research_conversation(self):
        # --- 1. Define and Process the Conversation ---
        conversation = [
            "My advisor, Dr. Anya Sharma, suggested I tailor my final project on distributed systems for the software engineering internship at Cloudflare. I think it's a great idea to get ahead.",
            "Honestly, I'm feeling pretty overwhelmed by the upcoming midterms, especially the one for Professor Evans's class on algorithms. That course is notoriously difficult and is a major stressor for me right now.",
            "I was thinking of organizing a study group session at the main library for this weekend. I already asked Michael and Jessica from my class if they wanted to join me on Saturday afternoon.",
            "To prepare for technical interviews, I've been grinding problems on LeetCode and trying to get better at Python. My goal is to eventually work at a company like Google or maybe even a smaller startup.",
            "Quick update on the robotics project: we finally got the new sensor module working yesterday. David is going to integrate it with the main codebase by the end of next week, which should put us back on schedule."
        ]

        for i, message in enumerate(conversation):

            # if i == 2:
            #     self._process_message(message)
            #     break
            
            self._process_message(message)


if __name__ == '__main__':
    unittest.main()
