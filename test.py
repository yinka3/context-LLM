import unittest
import logging
from typing import Optional
import logging_setup 
from context import Context 
from dtypes import MessageData, EntityData

logging_setup.setup_logging()

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

    def _get_entity_by_name(self, name: str) -> Optional[EntityData]:
        """Helper method to find an entity in the graph by its name or alias."""
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
            "The first draft for Project Starlight is due on October 15th, so we need to get started soon.",
        ]

        for message in conversation:
            self._process_message(message)

        # --- 2. Retrieve All Key Entities for Assertions ---
        # Using assertIsNotNone ensures the test fails early if a critical entity is missing.
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

        # NEW TEST: Check for possessive relationship from the new pattern
        # "Professor Davies' class"
        self.assertTrue(self.context.graph.has_edge(f"ent_{prof_davies.id}", f"ent_{prof_davies.id}"), 
                        "Should have an edge for possessive relationship.")
        
        edge_data = self.context.graph.get_edge_data(f"ent_{prof_davies.id}", f"ent_{prof_davies.id}")
        self.assertEqual(edge_data.get('relation'), 'has_possession_of')

        # UPDATED TEST: Adjectival modifiers are now low-confidence and NOT applied
        # In "powerful new technology", 'powerful' and 'new' are flagged for Tier 2
        # So the "helios_ai" entity should NOT have a 'has_property' attribute.
        self.assertNotIn("has_property", helios_ai.attributes, 
                         "Low-confidence attribute 'powerful' should not have been applied.")

        # TEST: Passive relationship and sourcing
        # "the Helios AI was built by their internal Core AI team"
        self.assertTrue(self.context.graph.has_edge(f"ent_{core_ai_team.id}", f"ent_{helios_ai.id}"),
                        "Core AI team should have a 'builds' relationship edge to Helios AI.")
        
        # TEST: High-confidence attribute extraction
        # "Project Starlight is very important..."
        self.assertIn("is_be", project_starlight.attributes, 
                      "Project Starlight should have an 'is_be' attribute.")
        self.assertEqual(project_starlight.attributes["is_be"][0].value, "very important",
                         "Project Starlight's importance attribute is incorrect.")
        
        # TEST: Coreference-enabled fact extraction
        # "Maria is going to work with me..."
        self.assertTrue(self.context.graph.has_edge(f"ent_{maria.id}", f"ent_{user_entity.id}"),
                        "Maria should have a 'works_with' relationship edge to the USER.")
        
        # TEST: High-confidence specific pattern
        # "...due on October 15th"
        self.assertIn("has_due_date", project_starlight.attributes,
                      "Project Starlight should have a due date.")
        self.assertEqual(project_starlight.attributes["has_due_date"][0].value, "October 15th")

        logging.info("All assertions passed successfully!")


if __name__ == '__main__':
    unittest.main()
