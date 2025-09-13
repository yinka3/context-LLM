import unittest
import logging
from typing import Optional
import logging_setup 
from context import Context 
from dtypes import MessageData, EntityData



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
            "Okay, I need to start planning my research paper for Professor Davies' class. The main topic I've chosen is a company called Innovate Dynamics.",
            "Innovate Dynamics has developed a powerful new technology known as the Helios AI. My research project, Project Starlight, will analyze the impact of this product.",
            "According to my sources, the Helios AI was built by their internal Core AI team. I also found that this entire team is located in Boston.",
            "Good news, my classmate Maria is going to work with me on Project Starlight. She knows a lot about machine learning, which is a key technology for the Helios AI.",
            "Professor Davies mentioned that Project Starlight is very important for my final grade. The first draft for Project Starlight is due on October 15th, so we need to get started soon.",
            "I'm feeling pretty stressed about my chemistry midterm, I think I'll ft Maria later to vent.",
            "My major is computer science and I want to learn Python. I'd love to get an internship at Google eventually.",
            "I just joined the AI club to meet some new people, since I'm trying to be more social this semester."
        ]

        for message in conversation:
            self._process_message(message)


if __name__ == '__main__':
    unittest.main()
