import unittest
import logging
from typing import Optional

from context import Context
from dtypes import MessageData, EntityData

logging.basicConfig(level=logging.WARNING)


class TestTier1ConversationFlow(unittest.TestCase):

    def setUp(self):
        self.context = Context()
        self.message_id_counter = 0

    def _process_message(self, msg: str):
        """Helper method to process a single message."""
        msg_data = MessageData(
            id=self.message_id_counter,
            role="user",
            message=msg,
            sentiment="unknown"
        )
        self.context.add(msg_data)
        self.message_id_counter += 1

    def _get_entity_by_name(self, name: str) -> Optional[EntityData]:
        search_name_lower = name.lower()
        for _, node_data in self.context.graph.nodes(data=True):
            if node_data.get("type") == "entity":
                entity = node_data["data"]
                if entity.name.lower() == search_name_lower:
                    return entity
                for alias in entity.aliases:
                    if alias['text'].lower() == search_name_lower:
                        return entity
        return None

    def test_student_research_conversation(self):
        # --- 1. Define and Process the Conversation ---
        conversation = [
            "Okay, I need to start planning my research paper for Professor Davies' class. The main topic I've chosen is a company called Innovate Dynamics.",
            "Innovate Dynamics has developed a powerful new technology known as the Helios AI. My research project, which I'm naming Project Starlight, will analyze the impact of this product.",
            "According to my sources, the Helios AI was built by their internal Core AI team. I also found that this entire team is located in Boston.",
            "Good news, my classmate Maria is going to work with me on Project Starlight. She knows a lot about machine learning, which is a key technology for the Helios AI.",
            "Professor Davies mentioned that Project Starlight is very important for my final grade.",
            "The most impressive part of the Helios AI is its predictive modeling feature. Innovate Dynamics uses this for all of their internal market analysis.",
            "The first draft for Project Starlight is due on October 15th, so we need to get started soon.",
            "I need to schedule a meeting with Maria tomorrow to discuss our plan for the first draft. It should be easy to coordinate since she lives near the main campus.",
            "I think this project will be challenging but rewarding. On a side note, the company is also planning a completely new initiative for next year."
        ]

        for message in conversation:
            self._process_message(message)

        # --- 2. Retrieve All Key Entities for Assertions ---
        # We use assertIsNotNone to fail the test early if a critical entity wasn't created.
        prof_davies = self._get_entity_by_name("Professor Davies")
        self.assertIsNotNone(prof_davies, "Entity 'Professor Davies' not found.")

        innovate_dynamics = self._get_entity_by_name("Innovate Dynamics")
        self.assertIsNotNone(innovate_dynamics, "Entity 'Innovate Dynamics' not found.")

        helios_ai = self._get_entity_by_name("Helios AI")
        self.assertIsNotNone(helios_ai, "Entity 'Helios AI' not found.")

        project_starlight = self._get_entity_by_name("Project Starlight")
        self.assertIsNotNone(project_starlight, "Entity 'Project Starlight' not found.")

        core_ai_team = self._get_entity_by_name("Core AI team")
        self.assertIsNotNone(core_ai_team, "Entity 'Core AI team' not found.")

        boston = self._get_entity_by_name("Boston")
        self.assertIsNotNone(boston, "Entity 'Boston' not found.")

        maria = self._get_entity_by_name("Maria")
        self.assertIsNotNone(maria, "Entity 'Maria' not found.")

        user_entity = self._get_entity_by_name("USER")
        self.assertIsNotNone(user_entity, "Canonical USER entity not found.")

        # --- 3. Perform Assertions on the Final Graph State ---

        # Test 1: Relationship Extraction
        self.assertIn("works_with", maria.attributes, "Maria should have a 'works_with' relationship.")
        self.assertEqual(maria.attributes["works_with"][0].value, user_entity, "Maria should work with the USER.")

        # Assert the SECOND fact: Maria works ON Project Starlight.
        self.assertIn("works_on", maria.attributes, "Maria should have a 'works_on' relationship.")
        self.assertEqual(maria.attributes["works_on"][0].value, project_starlight, "Maria should work on Project Starlight.")
        
        self.assertIn("is_located_in", core_ai_team.attributes, "Core AI team should have an 'is_located_in' relationship.")
        self.assertEqual(core_ai_team.attributes["is_located_in"][0].value, boston, "Core AI team should be located in Boston.")

        # Test 2: Multi-Fact Extraction from a Single Message
        # Both of these facts come from message #3
        self.assertIn("built", core_ai_team.attributes, "Core AI team should have a 'built' relationship.")
        self.assertEqual(core_ai_team.attributes["built"][0].value, helios_ai, "Core AI team should have built Helios AI.")

        # Test 3: Attribute/State Capture
        self.assertIn("is_be", project_starlight.attributes, "Project Starlight should have an 'is_be' attribute.")
        self.assertEqual(project_starlight.attributes["is_be"][0].value, "very important", "Project Starlight should be 'very important'.")

        # Test 4: Alias Enrichment
        # Check if the full noun chunk "the Helios AI" was added as an alias
        helios_aliases = [alias['text'].lower() for alias in helios_ai.aliases]
        self.assertIn("the helios ai", helios_aliases, "Helios AI should have 'the helios ai' as an alias.")

        # Test 5: Entity Merging
        # The system saw "Maria" twice. We check that her 'lives_near' attribute was added to the original entity.
        self.assertIn("lives_near", maria.attributes, "Maria should have a 'lives_near' attribute from the second mention.")


if __name__ == '__main__':
    unittest.main()