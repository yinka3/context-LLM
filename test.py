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
            "My machine learning course with Professor Hansen is getting intense. I need to start planning the final project.",
            "It's a very challenging project, but the topic, 'AI in robotics', is interesting.",
            "This kind of work is why I want to apply for the internship at Google. Their robotics division is world-class.",
            "I asked Sarah and David to work on the project with me. They seem excited about it.",
            "Honestly, I'm feeling anxious about the deadline. The whole thing was making me stressed.",
            "My last big report was written for a class at NYU. It was graded by the TA.",
            "Anyway, I'm going to the AI club meetup tonight. I hope to meet some new people.",
            "I saw Alex there and we talked for a bit. We grabbed coffee and discussed our classes. I felt much better after I talked with him. Socializing really helps with the stress.",
            "Today, Dr. Hansen gave me and Sarah some really useful feedback on our proposal.",
            "It wasn't all good though. I disagreed with David about the project's direction. He's more of an idealist. It's a classic conflict between the practical and the theoretical.",
            "I need to remember to ask the professor for a letter of recommendation for that Google internship.",
            "Our project team usually works in the main library. The building is always crowded.",
            "Managing a team, a difficult course, and my social life is complicated but rewarding."
        ]

        for message in conversation:
            self._process_message(message)


if __name__ == '__main__':
    unittest.main()
