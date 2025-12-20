"""
Vestige Integration Test Cases
Each case includes messages, expected entities, and validation criteria.
"""

# =============================================================================
# CASE 1: Long Messages (70-100 words, 2-4 entities each)
# Primary target: NLP extraction throughput, LLM context handling
# =============================================================================

LONG_MESSAGES = [
    """Had a really productive meeting with Marcus from the engineering team today. 
    We spent about two hours going over the new authentication system that's being 
    built for the Phoenix project. He mentioned that the deadline got pushed back 
    to March, which is a relief because the security audit from Deloitte isn't 
    complete yet. I need to follow up with him next week about the API documentation 
    he promised to send over.""",
    
    """Went to that new Italian place called Trattoria Bella on 5th Avenue with 
    my college roommate Derek last night. The pasta was incredible but the service 
    was slow. Derek's been working at Goldman Sachs for about three years now and 
    seems pretty burned out. He's thinking about taking a sabbatical to travel 
    through Southeast Asia. We talked about maybe doing a trip to Vietnam together 
    sometime next year if our schedules align.""",
    
    """Finally finished reading that book Dr. Patel recommended during my last 
    appointment. It's called Atomic Habits by James Clear and honestly it's changed 
    how I think about building routines. I've started implementing the two-minute 
    rule for my morning workouts. Planning to discuss my progress with Dr. Patel 
    when I see her next Thursday. She's been really helpful with my anxiety management 
    over the past few months.""",
    
    """The quarterly review at Nexus Technologies went better than expected. My 
    manager Linda gave positive feedback on the dashboard project I've been leading. 
    The CEO Richard Torres even stopped by our floor to congratulate the team. 
    There's talk of expanding the project scope to include the European markets, 
    which would mean collaborating with the London office. I might need to travel 
    there in April for the initial planning sessions.""",
    
    """Spent the weekend helping my brother Kevin move into his new apartment in 
    Brooklyn. His girlfriend Emma was there too and she's actually really nice 
    once you get to know her. The place is small but has great natural light and 
    is only two blocks from Prospect Park. Kevin's new job at the architecture 
    firm Gensler starts next month so the timing worked out perfectly. Mom is 
    already asking when she can visit to see the place.""",
]

EASY_REFERENCE_MESSAGES = [
    "Met Jake at the coffee shop this morning. He works at Stripe.",
    "Jake introduced me to his coworker Priya. She's on the payments team at Stripe.",
    "Had lunch with Jake and Priya near the Stripe office downtown.",
    "Jake mentioned he's been at Stripe for two years now.",
    "Priya told me she actually hired Jake when she was a team lead.",
    "Going to Jake's housewarming party next weekend. Priya will be there too.",
]

MEDIUM_REFERENCE_MESSAGES = [
    # --- BATCH 1 (1-5) : ESTABLISH BASELINE ---
    # TWEAK: Added "Junior Developer" to set up the promotion arc later.
    """My name is Jacob but everyone calls me Jake. I just started working as a 
    Junior Developer at a fintech startup called Meridian. My manager is 
    Benjamin Chen, though the team just calls him Ben.""",
    
    """Had my first one-on-one with Benjamin Chen today. He walked me through 
    Meridian's product roadmap for the next quarter. Really impressed with the 
    company's vision.""",
    
    """Ben assigned me to the authentication module. Jake is excited to finally 
    dig into some real code. The Meridian codebase is cleaner than I expected.""",
    
    """Lunch with Ben and a senior engineer named Sofia Rodriguez. Sofia has been 
    at Meridian for three years. She mentioned that Benjamin Chen recruited her 
    from Google.""",
    
    """Sofia goes by Sof with close friends. Sof and Jake grabbed coffee after 
    the standup. She gave me great advice about navigating the codebase.""",
    
    # --- BATCH 2 (6-10) : COMPLEXITY & NEW ENTITIES ---
    """Weekly sync with Ben, Sof, and the rest of the team. Jacob presented his 
    progress on the auth module. Benjamin seemed pleased with the direction.""",

    """Met the new Product Lead, Marcus Thorne, today. Marcus used to work with 
    Ben at a company called Nexus. He seems intense but knowledgeable about 
    fintech security.""",

    """Marcus pushed back on the timeline for the authentication module. He thinks 
    Jake needs more time to handle edge cases. Chen agreed and adjusted the sprint.""",

    """Sof invited Marcus to join us for drinks, but he declined. Sofia thinks 
    he is just stressed about the Meridian quarterly goals. She is always 
    looking out for team morale.""",

    """Late night deployment with Rodriguez. She saved me from pushing a bad 
    bug to production. I owe Sof a coffee tomorrow. The auth module is finally 
    stable.""",

    # --- BATCH 3 (11-15) : CONFLICT & LEADERSHIP ---
    """The CEO, Sarah Jenkins, stopped by the engineering floor. Sarah asked 
    Ben about the Nexus partnership rumors. Benjamin looked nervous.""",

    """Sarah Jenkins announced a town hall meeting for next Friday. Marcus Thorne 
    seems to be leading the presentation. Jake is worried it might impact the 
    roadmap.""",

    """Coffee with Ben. He admitted that Sarah is pushing for faster delivery. 
    Chen is trying to shield the engineering team from the pressure.""",

    """Sof and Marcus got into a heated debate about the API design. 
    Rodriguez argued for flexibility, while Thorne insisted on strict validation. 
    Jake stayed out of it.""",

    """The town hall went well. Sarah praised the engineering team for the 
    recent stability updates. Meridian stock options are vesting soon.""",

    # --- BATCH 4 (16-20) : THE "RETURN" (PROFILE UPDATE TRIGGER) ---
    """Performance review time. Benjamin Chen gave Jake a glowing review. 
    He specifically mentioned the work on the authentication module.""",

    """Jake got a promotion to Level 2 Engineer! Sof brought cupcakes to 
    celebrate. The whole Meridian team went out for dinner.""",

    """Reflecting on my time here. It's been crazy since I left Google, but 
    working with Ben and Sofia has been the best career move I've made.""", 

    """Marcus Thorne apologized to Sof for the argument earlier. They are 
    good now. Nexus is officially partnering with Meridian next month.""",

    """Ready for the next quarter. Sarah Jenkins outlined new goals. 
    Jake, Ben, and Sofia are ready to tackle the new payment gateway integration."""
]

GAME_STUDIO_MESSAGES = [
    # --- BATCH 1 (1-5): ESTABLISH BASELINE ---
    """My name is Elena Vasquez but the team calls me Lena. I just joined 
    Ironclad Games as a Lead Artist. My creative director is David Park.""",
    
    """First day working with David Park on the concept art for Project Titan. 
    He has a clear vision for the dark fantasy aesthetic we're going for.""",
    
    """David introduced me to the senior programmer, Rachel Kim. Rachel goes 
    by Rae around the office. She's been at Ironclad for five years.""",
    
    """Lunch with Rae and David. Rachel mentioned she was recruited from 
    Ubisoft Montreal. The Ironclad culture feels very different from big studios.""",
    
    """Rae showed me the game engine they built in-house called Forge. 
    Elena is impressed with how optimized it is for stylized rendering.""",
    
    # --- BATCH 2 (6-10): NEW ENTITIES + COMPLEXITY ---
    """Weekly art review with David, Rae, and the new narrative designer 
    Omar Hassan. Lena presented the character designs for the protagonist.""",
    
    """Met the studio head today, Victoria Chen. Victoria founded Ironclad 
    eight years ago after leaving Riot Games. She has high expectations.""",
    
    """Omar pitched a darker storyline for Project Titan. He thinks the 
    protagonist should have a morally ambiguous arc. David seemed hesitant.""",
    
    """Victoria stopped by to check on Project Titan's progress. She asked 
    David about the timeline for the vertical slice. Park looked stressed.""",
    
    """Late night session with Rae debugging a shader issue in Forge. 
    Rachel saved hours of work by finding the memory leak. I owe her coffee.""",
    
    # --- BATCH 3 (11-15): CONFLICT + PARTNERSHIPS ---
    """Victoria announced we're partnering with Ember Audio for the soundtrack. 
    Omar is excited because he worked with them at his previous studio.""",
    
    """Tension in the design meeting. Omar and David disagreed about the 
    combat system. Hassan wants more RPG elements, Park prefers action focus.""",
    
    """Coffee with Victoria. She admitted the studio is under pressure from 
    investors at Nexus Ventures. Chen is trying to protect the team from it.""",
    
    """Rae and Omar collaborated on the dialogue system integration. 
    Kim's technical skills combined with Hassan's writing made it seamless.""",
    
    """The vertical slice deadline is next month. David asked Lena to 
    prioritize the boss encounter visuals. Project Titan is coming together.""",
    
    # --- BATCH 4 (16-20): MILESTONE + CHANGES ---
    """Performance review with David Park. He praised my work on the 
    environment art. Elena got promoted to Art Director for Project Titan.""",
    
    """Celebrated my promotion with the team. Rae brought homemade cookies. 
    Victoria gave a speech about Ironclad's growth over the years.""",
    
    """Omar apologized to David for the heated debate last week. They 
    found a compromise on the combat design. The team feels more unified.""",
    
    """Ember Audio sent their lead composer, James Wright, to visit the 
    studio. James and Omar spent hours discussing the emotional beats.""",
    
    """The vertical slice presentation went well. Victoria showed it to 
    the Nexus Ventures partners. They approved additional funding.""",
    
    # --- BATCH 5 (21-25): EXPANSION + NEW CHALLENGES ---
    """Ironclad is hiring. David asked me to help interview candidates 
    for the junior artist position. We need more hands for Project Titan.""",
    
    """Hired a junior artist named Tyler Brooks. Tyler just graduated 
    from DigiPen. Lena is mentoring him on the Ironclad art pipeline.""",
    
    """Technical crisis. Forge crashed during a build and we lost two 
    days of work. Rae pulled an all-nighter to recover the assets.""",
    
    """Victoria announced a new project codenamed Spectre. She wants 
    Omar to lead the narrative team for it while staying on Titan.""",
    
    """James Wright from Ember Audio delivered the first soundtrack 
    samples. The main theme gave me chills. Perfect for Project Titan.""",
    
    # --- BATCH 6 (26-30): RESOLUTION + FUTURE ---
    """Tyler is settling in well. He and Rae bonded over their shared 
    love of retro games. Brooks has good instincts for pixel art.""",
    
    """Quarterly review meeting. Victoria praised the Project Titan team. 
    Chen mentioned potential acquisition talks but nothing confirmed.""",
    
    """David and I finalized the art bible for Project Titan. Park wants 
    to present it at the Game Developers Conference next spring.""",
    
    """Reflecting on my first six months at Ironclad. Working with David, 
    Rae, and Omar has been incredible. This team feels like family.""",
    
    """Victoria announced Project Titan will enter alpha next quarter. 
    Lena, David, Rae, Omar, and Tyler are ready for the final push."""
]

GYM_STUDENT_MESSAGES = [
    # --- BATCH 1 (1-5): ESTABLISH GYM BASELINE ---
    """My name is Jordan but my friends call me Jord. I've been going to 
    IronWorks Gym for about two months now. Finally got serious about fitness.""",
    
    """Met my personal trainer Marcus today. He's been working at IronWorks 
    for four years. Marcus put together a strength program for me.""",
    
    """First real session with Marcus. He pushed me hard on squats and 
    deadlifts. Marc said my form is decent but I need to work on hip mobility.""",
    
    """Ran into a girl named Destiny at the gym. She was using the squat rack 
    after me. Turns out she's also one of Marc's clients at IronWorks.""",
    
    """Destiny goes by Des. We grabbed smoothies after our workouts. 
    Des has been lifting for three years and competed in a powerlifting meet last summer.""",
    
    """Classes started this week. I'm taking Organic Chemistry with Professor 
    Okonkwo. Everyone warns that it's brutal but I need it for my major.""",
    
    """First lecture with Professor Okonkwo was intense. He moves fast but 
    explains concepts clearly. Dr. O said office hours are open anytime.""",
    
    """Told Marcus about my orgo class. Marc laughed and said chemistry almost 
    made him drop out. He suggested I find a study group early.""",
    
    """Prof Okon posted the first problem set. It's already overwhelming. 
    I need to find people to study with before I fall behind.""",
    
    """Met two people after lecture who want to form a study group. Priya 
    Sharma is pre-med and Caleb Morrison is a chem major. We're meeting Thursday.""",
    
    """First study session with Priya and Caleb. Priya is incredibly organized 
    and Caleb goes by Cal. We worked through the problem set together.""",
    
    """Had a gym session with Marcus. Told him about the study group. Marc introduced 
    me to another client named Tyler who's also a student at my university.""",
    
    """Tyler goes by Ty. He and Des are actually friends from high school. 
    Small world. Ty is majoring in kinesiology which explains the gym dedication.""",
    
    """Des, Ty, and I did a group workout today. Marcus programmed a 
    circuit for us. IronWorks felt more fun with friends around.""",
    
    """Prof Okon held a review session before the first quiz. Dr. O broke down 
    reaction mechanisms step by step. Priya took detailed notes for our group.""",
    
    # --- BATCH 4 (16-20): CLASS DIFFICULTY RAMPS UP ---
    """First quiz results came back. I got a B+ which feels like a win 
    in Professor Okonkwo's class. Cal got an A and Priya got a B.""",
    
    """Marcus noticed I've been stressed. Marc adjusted my program to include 
    more recovery work. He said overtraining won't help my grades.""",
    
    """Study session ran late. Caleb explained stereochemistry in a way that 
    finally clicked for me. Cal should be a TA honestly.""",
    
    """Ty asked if he could join our study group. Tyler is taking biochemistry 
    and said the orgo foundation would help. Priya welcomed him.""",
    
    """Dr. O announced the midterm will cover chapters one through six. 
    Professor Okonkwo said it's the hardest exam of the semester.""",
    
    # --- BATCH 5 (21-25): MIDTERM ARC + GYM AS STRESS RELIEF ---
    """Skipped the gym to study. Marcus texted to check in. Marc said even 
    twenty minutes of movement helps with retention. He's right.""",
    
    """Des noticed I've been absent from IronWorks. Destiny offered to do 
    morning sessions so I have evenings free for studying.""",
    
    """Morning workout with Des and Ty. Getting exercise before orgo lectures 
    actually helps me focus. Tyler suggested we make it a routine.""",
    
    """Priya is stressed about the midterm too. She and Cal came to IronWorks 
    with me. Caleb had never lifted before but Marcus gave them a quick intro.""",
    
    """Prof Okon's office hours were packed. I waited an hour but Dr. O 
    patiently answered everyone's questions. He genuinely wants us to succeed.""",
    
    # --- BATCH 6 (26-30): WORLDS COLLIDE ---
    """Had a study group session at my apartment. Ty brought snacks. Priya, Cal, Tyler, 
    and I worked through practice problems until midnight.""",
    
    """Marcus asked about my friends. Told him about Priya and Caleb. Marc 
    said he could set up a beginner program if they want to keep lifting.""",
    
    """Des met Priya at the gym today. Destiny and Priya hit it off immediately. 
    They're both into yoga and might take a class together.""",
    
    """Midterm tomorrow. Professor Okonkwo sent an encouraging email to the 
    class. Dr. O reminded us that one exam doesn't define us.""",
    
    """Midterm done. It was brutal but fair. Priya, Cal, Ty, and I went 
    to get food after. We're cautiously optimistic.""",
    
    # --- BATCH 7 (31-35): RESULTS + NEW GOALS ---
    """Midterm grades posted. I got a B+, Priya got an A-, Cal got an A, 
    and Tyler got a B. Professor Okonkwo curved it slightly.""",
    
    """Celebrated at IronWorks with a heavy lifting session. Marcus, Des, and 
    Ty all showed up. Marc said I've made serious progress since starting.""",
    
    """Priya officially signed up for sessions with Marcus. Destiny is helping 
    her learn the basics. The gym crew is expanding.""",
    
    """Prof Okon pulled me aside after lecture. Dr. O said my improvement has 
    been noticeable and encouraged me to consider research opportunities.""",
    
    """Cal suggested we keep the study group going for the final. Caleb wants 
    to aim for As across the board. Priya and Ty are in.""",
    
    # --- BATCH 8 (36-40): DEEPER CONNECTIONS ---
    """Had coffee with Marcus outside the gym. Marc opened up about his own college 
    struggles. He almost failed out before finding fitness.""",
    
    """Des and Ty are dating now. Destiny told me at the gym. She said 
    they've been friends so long it just made sense.""",
    
    """Priya invited the whole group to her family's Diwali celebration. 
    Cal, Ty, Des, and I all went. Her mom's cooking was incredible.""",
    
    """Professor Okonkwo hosted a department mixer. Dr. O introduced me to 
    some grad students doing research I'm interested in.""",
    
    """Had a late night gym session with just Marcus. Marc said I'm one of his 
    most dedicated clients. Means a lot coming from him.""",
    
    # --- BATCH 9 (41-45): REFLECTION + WRAP-UP ---
    """Reflecting on this semester. IronWorks went from intimidating to home. 
    Marcus, Des, and Ty have become real friends.""",
    
    """Final study session before the orgo exam. Priya, Cal, Tyler, and I 
    were at the library until closing. We know this material cold.""",
    
    """Final exam done. Prof Okon made it comprehensive but I felt prepared. 
    Cal thinks he aced it. Priya is just relieved it's over.""",
    
    """End of semester celebration at IronWorks. Marcus, Destiny, Tyler, Priya, 
    and Caleb all came. My worlds have completely merged.""",
    
    """Final grades in. I got an A- in Professor Okonkwo's class. Texted 
    Marc, Des, Ty, Priya, and Cal. Next semester we go even harder."""
]

FOUNDER_MESSAGES = [
    # --- BATCH 1 (1-5): ESTABLISH STARTUP CONTEXT ---
    """My name is Adeola but most people call me Deola or Ade. Finally taking the 
    leap and building Nexus AI full-time. Terrifying and exciting.""",
    
    """Met with my co-founder Kwame Asante today. Kwame and I have been 
    planning this for months. We're building developer tools for AI testing.""",
    
    """Working out of Civic Hub coworking space for now. Kwame found it — 
    decent wifi and free coffee. Can't complain at this stage.""",
    
    """Kwame and I spent all night on the pitch deck. Our value prop 
    is finally clicking: automated regression testing for LLM applications.""",
    
    """First real product decision today. Nexus will focus on API-first 
    approach. Kwame wants to prioritize the SDK but I think API comes first.""",
    
    # --- BATCH 2 (6-10): EARLY PRODUCT WORK ---
    """I'm exhausted but we shipped the first internal prototype. Nexus AI 
    is starting to feel real. Kwame handled the backend, I did the interface.""",
    
    """Got feedback from a friend who runs ML at a fintech. She said the 
    concept is solid but we need better documentation. Fair point.""",
    
    """Kwame Asante introduced me to his former colleague Yuki Tanaka. Yuki 
    has experience in developer relations and might advise us.""",
    
    """Call with Yuki went well. She gave us a framework for thinking about 
    developer experience. Yuki Tanaka knows her stuff.""",
    
    """Working on the landing page for Nexus. Kwame thinks we should launch 
    a waitlist before the product is ready. I'm hesitant but he might be right.""",
    
    # --- BATCH 3 (11-15): FUNDRAISING BEGINS ---
    """Had coffee with Samira Oyelaran today. She's an angel investor who 
    backed three dev tools companies. Samira asked tough questions.""",
    
    """Samira wants to see traction before committing. Fair enough. She 
    suggested we talk to Gradient Ventures — they lead pre-seed rounds.""",
    
    """I spent the weekend rewriting the pitch deck based on Samira's 
    feedback. Kwame added financial projections. We're getting sharper.""",
    
    """Intro call with Gradient Ventures went okay. The partner, David Park, 
    seemed interested but wants to see our technical architecture.""",
    
    """David Park from Gradient is coming to Civic Hub tomorrow for a deeper 
    dive. Kwame is preparing the technical demo. Nervous but ready.""",
    
    # --- BATCH 4 (16-20): INVESTOR MEETINGS ---
    """Gradient Ventures meeting done. David asked hard questions about 
    competitive moat. Kwame handled the technical parts brilliantly.""",
    
    """David Park wants to bring in his technical partner, Lisa Huang, for 
    a follow-up. Gradient VC is known for thorough due diligence.""",
    
    """While waiting on Gradient, Samira Oyelaran offered to invest 50k as 
    a bridge. She believes in us. That means a lot right now.""",
    
    """Kwame and I debated whether to take Samira's angel check or wait 
    for Gradient. Decided to take it — runway matters more than optics.""",
    
    """Signed the angel agreement with Samira today. First external money 
    in Nexus AI. Kwame and I celebrated with cheap champagne at Civic Hub.""",
    
    # --- BATCH 5 (21-25): PRODUCT GRIND ---
    """Back to building. Nexus needs a lot of work before we can show 
    Gradient's technical team. Kwame is refactoring the core engine.""",
    
    """Yuki Tanaka connected us with a potential beta customer — a startup 
    called Streamline Labs. They're building AI assistants for support teams.""",
    
    """Call with Streamline Labs went great. Their CTO, Omar Hassan, wants 
    to pilot Nexus for their testing pipeline. First real customer interest.""",
    
    """Omar from Streamline is technical and asked detailed questions. 
    Kwame Asante joined the call and they nerded out about architecture.""",
    
    """Sent Streamline Labs a pilot proposal. Omar Hassan said he'll review 
    with his team. Trying not to get too excited but this could be huge.""",
    
    # --- BATCH 6 (26-30): TECHNICAL CHALLENGES ---
    """Hit a major bug in the Nexus core. Our test generation is producing 
    false positives. Kwame has been debugging for two days straight.""",
    
    """Kwame finally found the issue — a race condition in the async 
    handlers. I helped with the fix. Partnership working well.""",
    
    """Lisa Huang from Gradient Ventures did the technical deep dive today. 
    She grilled us on scalability. Tough but fair questions.""",
    
    """Lisa seemed impressed with our architecture decisions. David Park 
    emailed saying Gradient is moving us to partner meeting stage.""",
    
    """Yuki Tanaka offered to join as an official advisor. Equity-based, 
    no cash. Kwame and I agreed immediately. Her network is invaluable.""",
    
    # --- BATCH 7 (31-35): MOMENTUM BUILDING ---
    """Partner meeting at Gradient Ventures next week. David Park said 
    it's the final step before term sheet. Nexus AI might actually get funded.""",
    
    """Prepping for Gradient partner meeting. Samira Oyelaran did a mock 
    pitch session with us. She didn't hold back on criticism. Needed that.""",
    
    """Omar Hassan confirmed Streamline Labs wants to move forward with 
    the pilot. First paying customer for Nexus. Small deal but validates us.""",
    
    """Signed pilot agreement with Streamline. 5k for three months. Omar 
    and Kwame already scheduling integration calls. It's happening.""",
    
    """I'm running on fumes but the momentum is real. Gradient meeting 
    in three days. Civic Hub feels like home now.""",
    
    # --- BATCH 8 (36-40): TERM SHEET ---
    """Gradient Ventures partner meeting done. Five partners in the room. 
    David Park and Lisa Huang championed us. Felt like an exam.""",
    
    """Got the call from David. Gradient Ventures is offering a term sheet. 
    1.5M pre-seed at 8M post. Kwame and I are in shock.""",
    
    """Samira Oyelaran is thrilled about the Gradient term sheet. Her angel 
    check converts on good terms. She's been such a champion for us.""",
    
    """Reviewing the term sheet with a lawyer Yuki Tanaka recommended. 
    Standard terms but want to make sure we understand everything.""",
    
    """Kwame Asante and I talked through the implications of the raise. 
    We're aligned on using it for hiring and infrastructure.""",
    
    # --- BATCH 9 (41-45): HIRING PHASE ---
    """Started recruiting now that funding is nearly closed. Need a senior 
    engineer badly. Kwame is stretched too thin on technical work.""",
    
    """Interviewed a candidate named Priya Mehta today. Strong backend 
    experience, worked at two YC companies. Kwame was impressed.""",
    
    """Second round with Priya Mehta. She asked great questions about 
    culture and equity. Feels like a fit for Nexus AI.""",
    
    """Priya accepted our offer. She starts in two weeks. First hire for 
    Nexus besides me and Kwame. Team of three now.""",
    
    """Omar Hassan from Streamline gave us a glowing reference for Priya. 
    Turns out they worked together briefly. Small world.""",
    
    # --- BATCH 10 (46-50): SCALING PAINS ---
    """Priya Mehta started today. Kwame is doing onboarding while I handle 
    investor paperwork. Lots of context to transfer.""",
    
    """Streamline Labs pilot is going well. Omar wants to expand scope 
    and budget. Nexus AI might have a real revenue path.""",
    
    """Growing pains with three people. Communication overhead is real. 
    Kwame suggested daily standups. I agreed reluctantly.""",
    
    """Gradient Ventures funding officially closed. Money hit the account. 
    David Park sent a welcome note to the portfolio.""",
    
    """Celebrated the close with the whole team at Civic Hub. Samira 
    Oyelaran stopped by. Yuki sent champagne. Feeling grateful.""",
    
    # --- BATCH 11 (51-55): PRODUCT EXPANSION ---
    """Now that we're funded, strategy session with Kwame and Priya. 
    Nexus needs to decide: go deeper on testing or expand to monitoring.""",
    
    """Priya Mehta made a compelling case for monitoring features. Her 
    experience at previous startups shaped her perspective.""",
    
    """Omar Hassan at Streamline said they'd pay more for monitoring. 
    Customer signal is clear. Kwame is designing the architecture.""",
    
    """Lisa Huang from Gradient checked in. She wants quarterly updates 
    and offered to intro us to other portfolio companies as customers.""",
    
    """David Park invited me to speak at a Gradient Ventures founder 
    event. Good visibility but I hate public speaking. Said yes anyway.""",
    
    # --- BATCH 12 (56-60): NEW CHAPTER ---
    """Spoke at the Gradient event. Terrifying but went okay. Met other 
    founders in the portfolio. Samira Oyelaran was in the audience cheering.""",
    
    """Three months since launch. Nexus AI has four paying customers now. 
    Small revenue but trending up. Kwame and Priya are crushing it.""",
    
    """Yuki Tanaka introduced us to a potential enterprise customer — a 
    bank doing AI compliance testing. Huge if it works out.""",
    
    """Reflecting on the journey so far. I started with just an idea. 
    Now there's a team, investors, and customers who believe in Nexus.""",
    
    """Planning session for next quarter. Kwame, Priya, and I aligned 
    on goals: hit 10 customers, ship monitoring, and start Series A prep."""
]

ALEX_MESSAGES = [
    "Hey, I'm Alex Chen. Junior at Boston University studying Computer Science. My advisor is Professor Williams but everyone calls her Dr. W. Living with my roommate Derek this year.",
    "Had my first Systems Programming lecture today with Professor Ramirez. He's intense but Derek says he's the best in the department.",
    "Study group forming for Systems. Met Jasmine and this guy Kevin at the library. Jas is a senior, super helpful already.",
    "Dr. W wants me to consider grad school. She mentioned her colleague at MIT, Professor Chen. Awkward sharing a last name honestly.",
    "Derek and I hit the gym at FitRec. He's training for a marathon so I just did weights while he ran forever.",
    "Kevin from study group is actually Kev from my freshman dorm. Small world. He didn't recognize me at first either.",
    "Prof Ramirez assigned the first project. Jasmine suggested we partner up. Jas has done systems stuff at her internship.",
    "Mom called asking about Thanksgiving plans. Told her Derek might come since his family is in California.",
    "Office hours with Dr. W today. She reviewed my resume for the Google internship. Said to mention Professor Ramirez as a reference.",
    "FitRec was packed so Derek and I tried the smaller gym in West Campus. Way better honestly.",
    "Jas and I finished the first milestone. Professor Ramirez said our approach was clever. Kevin is struggling with his partner.",
    "Met Derek's girlfriend finally. Her name is Sophie, she's in the nursing program. Really nice actually.",
    "Dr. Williams connected me with an alum named Marcus Thompson. He works at Stripe and offered to do a mock interview.",
    "Study session ran late. Kev brought coffee for everyone. Jasmine had to leave early for her research lab.",
    "Prof Ramirez extended the deadline after half the class bombed the midterm. I got a B+ somehow.",
    "Sophie invited me to her nursing school formal as Derek's plus one's plus one. Weird but sure.",
    "Mock interview with Marcus went well. He said my systems knowledge is solid thanks to Ramirez's class.",
    "Thanksgiving at home. Mom made too much food as usual. Showed her my project and she pretended to understand.",
    "Back at BU. Derek brought leftovers from Sophie's family too. We ate well this week.",
    "Final stretch of semester. Dr. W said she's proud of my progress. Jasmine and I are aiming for an A in Systems."
]