"""
Comprehensive NER Span Extraction Test

Tests RuleChef with 100 diverse training examples and 30 held-out test examples
to evaluate span extraction quality across domains and edge cases.
"""

import os
from openai import OpenAI
from rulechef import RuleChef, Task
from rulechef.coordinator import SimpleCoordinator
from rulechef.evaluation import evaluate_spans

# =============================================================================
# Define NER Task
# =============================================================================

task = Task(
    name="Named Entity Recognition",
    description="Extract named entities (people, organizations, locations, products) from text",
    input_schema={"text": "str"},
    output_schema={"spans": "List[Span]"},
)

# Check for API key
api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    print("⚠️  OPENAI_API_KEY not set - running in simulation mode")
    print("Set it with: export OPENAI_API_KEY='your-key'")
    chef = RuleChef(task, dataset_name="ner_comprehensive", allowed_formats=["regex"])
else:
    client = OpenAI(api_key=api_key, base_url="https://api.groq.com/openai/v1/")
    coordinator = SimpleCoordinator(trigger_threshold=20, verbose=False)
    chef = RuleChef(
        task,
        client=client,
        dataset_name="ner_comprehensive",
        coordinator=coordinator,
        auto_trigger=False,
        model="moonshotai/kimi-k2-instruct-0905",
    )

# =============================================================================
# Training Data (100 diverse examples)
# =============================================================================

training_examples = [
    # TECH COMPANIES & PRODUCTS (10 examples)
    {
        "text": "Apple Inc. announced its new iPhone 15 at an event in Cupertino, California.",
        "entities": [
            {"text": "Apple Inc.", "start": 0, "end": 10},
            {"text": "iPhone 15", "start": 33, "end": 42},
            {"text": "Cupertino", "start": 59, "end": 68},
            {"text": "California", "start": 70, "end": 80},
        ],
    },
    {
        "text": "Microsoft CEO Satya Nadella discussed partnerships at the Seattle headquarters.",
        "entities": [
            {"text": "Microsoft", "start": 0, "end": 9},
            {"text": "Satya Nadella", "start": 14, "end": 27},
            {"text": "Seattle", "start": 55, "end": 62},
        ],
    },
    {
        "text": "Google LLC and OpenAI signed a major collaboration deal in San Francisco.",
        "entities": [
            {"text": "Google LLC", "start": 0, "end": 10},
            {"text": "OpenAI", "start": 15, "end": 21},
            {"text": "San Francisco", "start": 57, "end": 70},
        ],
    },
    {
        "text": "Tesla Motors released the Model 3 with advanced AI features for autonomous driving.",
        "entities": [
            {"text": "Tesla Motors", "start": 0, "end": 12},
            {"text": "Model 3", "start": 25, "end": 32},
        ],
    },
    {
        "text": "Amazon Web Services expanded cloud infrastructure in Tokyo and Singapore.",
        "entities": [
            {"text": "Amazon Web Services", "start": 0, "end": 19},
            {"text": "Tokyo", "start": 57, "end": 62},
            {"text": "Singapore", "start": 67, "end": 76},
        ],
    },
    {
        "text": "Netflix Inc. produced content at studios in Mumbai, India and Los Angeles, California.",
        "entities": [
            {"text": "Netflix Inc.", "start": 0, "end": 12},
            {"text": "Mumbai", "start": 43, "end": 49},
            {"text": "India", "start": 51, "end": 56},
            {"text": "Los Angeles", "start": 61, "end": 72},
            {"text": "California", "start": 74, "end": 84},
        ],
    },
    {
        "text": "Meta Platforms (formerly Facebook) opened AI research lab in Dublin, United Kingdom.",
        "entities": [
            {"text": "Meta Platforms", "start": 0, "end": 14},
            {"text": "Facebook", "start": 25, "end": 33},
            {"text": "Dublin", "start": 62, "end": 68},
            {"text": "United Kingdom", "start": 70, "end": 84},
        ],
    },
    {
        "text": "Nvidia Corporation showcased GPUs at the conference in Las Vegas, Nevada.",
        "entities": [
            {"text": "Nvidia Corporation", "start": 0, "end": 17},
            {"text": "Las Vegas", "start": 53, "end": 62},
            {"text": "Nevada", "start": 64, "end": 70},
        ],
    },
    {
        "text": "IBM partnered with Red Hat on cloud solutions from Boston headquarters.",
        "entities": [
            {"text": "IBM", "start": 0, "end": 3},
            {"text": "Red Hat", "start": 18, "end": 25},
            {"text": "Boston", "start": 54, "end": 60},
        ],
    },
    {
        "text": "Intel Corporation is building a new fab plant in Phoenix, Arizona.",
        "entities": [
            {"text": "Intel Corporation", "start": 0, "end": 16},
            {"text": "Phoenix", "start": 48, "end": 55},
            {"text": "Arizona", "start": 57, "end": 64},
        ],
    },
    # FINANCE & BUSINESS (10 examples)
    {
        "text": "Goldman Sachs reported quarterly earnings from New York City headquarters.",
        "entities": [
            {"text": "Goldman Sachs", "start": 0, "end": 13},
            {"text": "New York City", "start": 45, "end": 58},
        ],
    },
    {
        "text": "JPMorgan Chase announced merger plans with Bank of America.",
        "entities": [
            {"text": "JPMorgan Chase", "start": 0, "end": 14},
            {"text": "Bank of America", "start": 42, "end": 57},
        ],
    },
    {
        "text": "Warren Buffett, CEO of Berkshire Hathaway, spoke at shareholders meeting.",
        "entities": [
            {"text": "Warren Buffett", "start": 0, "end": 14},
            {"text": "Berkshire Hathaway", "start": 23, "end": 41},
        ],
    },
    {
        "text": "The Wall Street Journal reported on Fed policy changes announced by Jerome Powell.",
        "entities": [
            {"text": "Wall Street Journal", "start": 4, "end": 22},
            {"text": "Fed", "start": 39, "end": 42},
            {"text": "Jerome Powell", "start": 67, "end": 80},
        ],
    },
    {
        "text": "Coca-Cola Company expanded operations across Europe and Asia-Pacific.",
        "entities": [
            {"text": "Coca-Cola Company", "start": 0, "end": 17},
            {"text": "Europe", "start": 44, "end": 50},
            {"text": "Asia-Pacific", "start": 55, "end": 67},
        ],
    },
    {
        "text": "Elon Musk founded companies including Tesla, SpaceX, and Neuralink.",
        "entities": [
            {"text": "Elon Musk", "start": 0, "end": 9},
            {"text": "Tesla", "start": 38, "end": 43},
            {"text": "SpaceX", "start": 45, "end": 51},
            {"text": "Neuralink", "start": 57, "end": 66},
        ],
    },
    {
        "text": "Microsoft CEO Satya Nadella partnered with OpenAI for enterprise AI.",
        "entities": [
            {"text": "Microsoft", "start": 0, "end": 9},
            {"text": "Satya Nadella", "start": 14, "end": 27},
            {"text": "OpenAI", "start": 44, "end": 50},
        ],
    },
    {
        "text": "Mark Zuckerberg announced Meta's metaverse strategy at developer conference.",
        "entities": [
            {"text": "Mark Zuckerberg", "start": 0, "end": 15},
            {"text": "Meta", "start": 26, "end": 30},
        ],
    },
    {
        "text": "Amazon founder Jeff Bezos stepped down to focus on Blue Origin space ventures.",
        "entities": [
            {"text": "Jeff Bezos", "start": 17, "end": 27},
            {"text": "Amazon", "start": 0, "end": 6},
            {"text": "Blue Origin", "start": 52, "end": 63},
        ],
    },
    # MEDICAL & SCIENCE (10 examples)
    {
        "text": "Johns Hopkins University research team published study on COVID-19 variants.",
        "entities": [
            {"text": "Johns Hopkins University", "start": 0, "end": 23},
            {"text": "COVID-19", "start": 56, "end": 64},
        ],
    },
    {
        "text": "Dr. Anthony Fauci at National Institute of Allergy and Infectious Diseases reported findings.",
        "entities": [
            {"text": "Anthony Fauci", "start": 4, "end": 17},
            {
                "text": "National Institute of Allergy and Infectious Diseases",
                "start": 21,
                "end": 75,
            },
        ],
    },
    {
        "text": "Harvard Medical School and MIT collaborated on genome sequencing project.",
        "entities": [
            {"text": "Harvard Medical School", "start": 0, "end": 21},
            {"text": "MIT", "start": 26, "end": 29},
        ],
    },
    {
        "text": "Stanford University bioengineers developed new immunotherapy at Palo Alto facility.",
        "entities": [
            {"text": "Stanford University", "start": 0, "end": 19},
            {"text": "Palo Alto", "start": 64, "end": 73},
        ],
    },
    {
        "text": "The Lancet published breakthrough research from University of Oxford.",
        "entities": [
            {"text": "Lancet", "start": 4, "end": 10},
            {"text": "University of Oxford", "start": 43, "end": 63},
        ],
    },
    {
        "text": "WHO announced new guidelines for Omicron variant management.",
        "entities": [
            {"text": "WHO", "start": 0, "end": 3},
            {"text": "Omicron", "start": 33, "end": 40},
        ],
    },
    {
        "text": "Pfizer and BioNTech partnered on vaccine development at facilities in Michigan.",
        "entities": [
            {"text": "Pfizer", "start": 0, "end": 6},
            {"text": "BioNTech", "start": 11, "end": 19},
            {"text": "Michigan", "start": 71, "end": 79},
        ],
    },
    {
        "text": "UC San Diego researchers at Salk Institute discovered breakthrough in regenerative medicine.",
        "entities": [
            {"text": "UC San Diego", "start": 0, "end": 12},
            {"text": "Salk Institute", "start": 27, "end": 40},
        ],
    },
    {
        "text": "Yale University School of Medicine collaborated with UC Berkeley on study.",
        "entities": [
            {"text": "Yale University School of Medicine", "start": 0, "end": 33},
            {"text": "UC Berkeley", "start": 54, "end": 65},
        ],
    },
    {
        "text": "Nature journal featured research from Caltech in Pasadena, California.",
        "entities": [
            {"text": "Nature", "start": 0, "end": 6},
            {"text": "Caltech", "start": 37, "end": 44},
            {"text": "Pasadena", "start": 48, "end": 56},
            {"text": "California", "start": 58, "end": 68},
        ],
    },
    # GOVERNMENT & POLITICS (10 examples)
    {
        "text": "U.S. Department of State announced new diplomatic policy regarding China.",
        "entities": [
            {"text": "U.S. Department of State", "start": 0, "end": 24},
            {"text": "China", "start": 68, "end": 73},
        ],
    },
    {
        "text": "President Biden met with Prime Minister Sunak at the White House.",
        "entities": [
            {"text": "Biden", "start": 10, "end": 15},
            {"text": "Prime Minister Sunak", "start": 24, "end": 44},
            {"text": "White House", "start": 52, "end": 63},
        ],
    },
    {
        "text": "The British Parliament voted on legislation in London concerning trade.",
        "entities": [
            {"text": "British Parliament", "start": 4, "end": 21},
            {"text": "London", "start": 45, "end": 51},
        ],
    },
    {
        "text": "U.N. Secretary General António Guterres addressed climate crisis at United Nations headquarters.",
        "entities": [
            {"text": "U.N. Secretary General", "start": 0, "end": 21},
            {"text": "António Guterres", "start": 22, "end": 37},
            {"text": "United Nations", "start": 70, "end": 84},
        ],
    },
    {
        "text": "German Chancellor Angela Merkel visited Brussels for EU summit meeting.",
        "entities": [
            {"text": "German Chancellor", "start": 0, "end": 16},
            {"text": "Angela Merkel", "start": 17, "end": 30},
            {"text": "Brussels", "start": 38, "end": 46},
            {"text": "EU", "start": 51, "end": 53},
        ],
    },
    {
        "text": "The Senate passed bill with support from both Republican and Democratic parties.",
        "entities": [
            {"text": "Senate", "start": 4, "end": 10},
            {"text": "Republican", "start": 48, "end": 58},
            {"text": "Democratic", "start": 63, "end": 73},
        ],
    },
    {
        "text": "Tokyo hosted G20 summit with leaders from Japan, China, and United States.",
        "entities": [
            {"text": "Tokyo", "start": 0, "end": 5},
            {"text": "G20", "start": 12, "end": 15},
            {"text": "Japan", "start": 39, "end": 44},
            {"text": "China", "start": 46, "end": 51},
            {"text": "United States", "start": 57, "end": 70},
        ],
    },
    {
        "text": "NATO summit in Brussels addressed security concerns regarding Russia.",
        "entities": [
            {"text": "NATO", "start": 0, "end": 4},
            {"text": "Brussels", "start": 15, "end": 23},
            {"text": "Russia", "start": 62, "end": 68},
        ],
    },
    {
        "text": "Israeli Prime Minister Benjamin Netanyahu met with Egyptian counterpart.",
        "entities": [
            {"text": "Israeli", "start": 0, "end": 7},
            {"text": "Benjamin Netanyahu", "start": 24, "end": 41},
            {"text": "Egyptian", "start": 50, "end": 58},
        ],
    },
    # SPORTS (10 examples)
    {
        "text": "Tom Brady played for New England Patriots and Tampa Bay Buccaneers.",
        "entities": [
            {"text": "Tom Brady", "start": 0, "end": 9},
            {"text": "New England Patriots", "start": 24, "end": 43},
            {"text": "Tampa Bay Buccaneers", "start": 48, "end": 67},
        ],
    },
    {
        "text": "Cristiano Ronaldo transferred to Manchester United from Real Madrid.",
        "entities": [
            {"text": "Cristiano Ronaldo", "start": 0, "end": 17},
            {"text": "Manchester United", "start": 33, "end": 50},
            {"text": "Real Madrid", "start": 56, "end": 67},
        ],
    },
    {
        "text": "Serena Williams won Wimbledon Championships in London, England.",
        "entities": [
            {"text": "Serena Williams", "start": 0, "end": 14},
            {"text": "Wimbledon Championships", "start": 19, "end": 41},
            {"text": "London", "start": 45, "end": 51},
            {"text": "England", "start": 53, "end": 60},
        ],
    },
    {
        "text": "LeBron James led Lakers to championship at Staples Center in Los Angeles.",
        "entities": [
            {"text": "LeBron James", "start": 0, "end": 12},
            {"text": "Lakers", "start": 21, "end": 27},
            {"text": "Staples Center", "start": 44, "end": 58},
            {"text": "Los Angeles", "start": 62, "end": 73},
        ],
    },
    {
        "text": "Lionel Messi joined Paris Saint-Germain after leaving FC Barcelona.",
        "entities": [
            {"text": "Lionel Messi", "start": 0, "end": 12},
            {"text": "Paris Saint-Germain", "start": 20, "end": 39},
            {"text": "FC Barcelona", "start": 54, "end": 66},
        ],
    },
    {
        "text": "Jordan Brand sponsored athlete performances at Olympics in Tokyo.",
        "entities": [
            {"text": "Jordan Brand", "start": 0, "end": 12},
            {"text": "Olympics", "start": 40, "end": 48},
            {"text": "Tokyo", "start": 52, "end": 57},
        ],
    },
    {
        "text": "Roger Federer retired from professional tennis after competing at Wimbledon.",
        "entities": [
            {"text": "Roger Federer", "start": 0, "end": 13},
            {"text": "Wimbledon", "start": 67, "end": 76},
        ],
    },
    {
        "text": "Naomi Osaka withdrew from French Open tournament in Paris.",
        "entities": [
            {"text": "Naomi Osaka", "start": 0, "end": 11},
            {"text": "French Open", "start": 25, "end": 36},
            {"text": "Paris", "start": 50, "end": 55},
        ],
    },
    {
        "text": "Neymar plays for PSG in Paris, France with other professional athletes.",
        "entities": [
            {"text": "Neymar", "start": 0, "end": 6},
            {"text": "PSG", "start": 20, "end": 23},
            {"text": "Paris", "start": 27, "end": 32},
            {"text": "France", "start": 34, "end": 40},
        ],
    },
    # ENTERTAINMENT & MEDIA (10 examples)
    {
        "text": "Steven Spielberg directed blockbuster at Universal Studios in Los Angeles.",
        "entities": [
            {"text": "Steven Spielberg", "start": 0, "end": 16},
            {"text": "Universal Studios", "start": 36, "end": 52},
            {"text": "Los Angeles", "start": 56, "end": 67},
        ],
    },
    {
        "text": "Taylor Swift released album at Spotify and Apple Music.",
        "entities": [
            {"text": "Taylor Swift", "start": 0, "end": 12},
            {"text": "Spotify", "start": 30, "end": 37},
            {"text": "Apple Music", "start": 42, "end": 53},
        ],
    },
    {
        "text": "HBO aired Game of Thrones series produced in Northern Ireland.",
        "entities": [
            {"text": "HBO", "start": 0, "end": 3},
            {"text": "Game of Thrones", "start": 9, "end": 24},
            {"text": "Northern Ireland", "start": 44, "end": 60},
        ],
    },
    {
        "text": "Disney+ released Marvel series with actors from United States.",
        "entities": [
            {"text": "Disney+", "start": 0, "end": 7},
            {"text": "Marvel", "start": 17, "end": 23},
            {"text": "United States", "start": 48, "end": 61},
        ],
    },
    {
        "text": "Avatar 2 directed by James Cameron released by 20th Century Studios.",
        "entities": [
            {"text": "Avatar 2", "start": 0, "end": 8},
            {"text": "James Cameron", "start": 20, "end": 33},
            {"text": "20th Century Studios", "start": 45, "end": 64},
        ],
    },
    {
        "text": "Beyoncé performed at Coachella music festival in California.",
        "entities": [
            {"text": "Beyoncé", "start": 0, "end": 7},
            {"text": "Coachella", "start": 20, "end": 29},
            {"text": "California", "start": 48, "end": 58},
        ],
    },
    {
        "text": "Netflix released Wednesday series with actress Jenna Ortega.",
        "entities": [
            {"text": "Netflix", "start": 0, "end": 7},
            {"text": "Wednesday", "start": 17, "end": 26},
            {"text": "Jenna Ortega", "start": 45, "end": 57},
        ],
    },
    {
        "text": "The New York Times reported on entertainment industry changes.",
        "entities": [
            {"text": "New York Times", "start": 4, "end": 18},
        ],
    },
    {
        "text": "Amazon Prime Video signed exclusive deal with A24 production company.",
        "entities": [
            {"text": "Amazon Prime Video", "start": 0, "end": 17},
            {"text": "A24", "start": 44, "end": 47},
        ],
    },
    # LOCATIONS & TRAVEL (10 examples)
    {
        "text": "Tourists visited Eiffel Tower in Paris, France.",
        "entities": [
            {"text": "Eiffel Tower", "start": 17, "end": 29},
            {"text": "Paris", "start": 33, "end": 38},
            {"text": "France", "start": 40, "end": 46},
        ],
    },
    {
        "text": "Big Ben and Houses of Parliament located in London, United Kingdom.",
        "entities": [
            {"text": "Big Ben", "start": 0, "end": 7},
            {"text": "Houses of Parliament", "start": 12, "end": 31},
            {"text": "London", "start": 42, "end": 48},
            {"text": "United Kingdom", "start": 50, "end": 64},
        ],
    },
    {
        "text": "Great Wall of China spans across northern China.",
        "entities": [
            {"text": "Great Wall of China", "start": 0, "end": 19},
            {"text": "China", "start": 40, "end": 45},
        ],
    },
    {
        "text": "Grand Canyon located in Arizona near Las Vegas, Nevada.",
        "entities": [
            {"text": "Grand Canyon", "start": 0, "end": 11},
            {"text": "Arizona", "start": 25, "end": 32},
            {"text": "Las Vegas", "start": 39, "end": 48},
            {"text": "Nevada", "start": 50, "end": 56},
        ],
    },
    {
        "text": "Statue of Liberty in New York City overlooks Hudson River.",
        "entities": [
            {"text": "Statue of Liberty", "start": 0, "end": 17},
            {"text": "New York City", "start": 21, "end": 34},
            {"text": "Hudson River", "start": 44, "end": 56},
        ],
    },
    {
        "text": "Golden Gate Bridge connects San Francisco to Marin County, California.",
        "entities": [
            {"text": "Golden Gate Bridge", "start": 0, "end": 17},
            {"text": "San Francisco", "start": 29, "end": 42},
            {"text": "Marin County", "start": 46, "end": 58},
            {"text": "California", "start": 60, "end": 70},
        ],
    },
    {
        "text": "Mount Everest peak located in Himalayas between Nepal and Tibet.",
        "entities": [
            {"text": "Mount Everest", "start": 0, "end": 13},
            {"text": "Himalayas", "start": 30, "end": 39},
            {"text": "Nepal", "start": 48, "end": 53},
            {"text": "Tibet", "start": 58, "end": 63},
        ],
    },
    {
        "text": "Amazon Rainforest spans across Brazil, Peru, and Ecuador.",
        "entities": [
            {"text": "Amazon Rainforest", "start": 0, "end": 17},
            {"text": "Brazil", "start": 33, "end": 39},
            {"text": "Peru", "start": 41, "end": 45},
            {"text": "Ecuador", "start": 51, "end": 58},
        ],
    },
    {
        "text": "Dead Sea located between Israel, Palestine, and Jordan.",
        "entities": [
            {"text": "Dead Sea", "start": 0, "end": 8},
            {"text": "Israel", "start": 20, "end": 26},
            {"text": "Palestine", "start": 28, "end": 37},
            {"text": "Jordan", "start": 43, "end": 49},
        ],
    },
    {
        "text": "Sahara Desert extends across North Africa including Egypt and Morocco.",
        "entities": [
            {"text": "Sahara Desert", "start": 0, "end": 13},
            {"text": "North Africa", "start": 29, "end": 41},
            {"text": "Egypt", "start": 52, "end": 57},
            {"text": "Morocco", "start": 62, "end": 69},
        ],
    },
    # EDGE CASES - MULTI-WORD & AMBIGUOUS (10 examples)
    {
        "text": "University of California Berkeley established in San Francisco Bay area.",
        "entities": [
            {"text": "University of California Berkeley", "start": 0, "end": 33},
            {"text": "San Francisco Bay", "start": 48, "end": 65},
        ],
    },
    {
        "text": "New York University located in Manhattan, New York City.",
        "entities": [
            {"text": "New York University", "start": 0, "end": 19},
            {"text": "Manhattan", "start": 32, "end": 41},
            {"text": "New York City", "start": 43, "end": 56},
        ],
    },
    {
        "text": "Washington D.C. and Washington State are both named after George Washington.",
        "entities": [
            {"text": "Washington D.C.", "start": 0, "end": 15},
            {"text": "Washington State", "start": 20, "end": 35},
            {"text": "George Washington", "start": 59, "end": 76},
        ],
    },
    {
        "text": "Apple Inc. headquarters in Cupertino versus apple pie recipe.",
        "entities": [
            {"text": "Apple Inc.", "start": 0, "end": 10},
            {"text": "Cupertino", "start": 26, "end": 35},
        ],
    },
    {
        "text": "Amazon rainforest and Amazon.com headquarters both known worldwide.",
        "entities": [
            {"text": "Amazon rainforest", "start": 0, "end": 17},
            {"text": "Amazon.com", "start": 22, "end": 32},
        ],
    },
    {
        "text": "Crown Princess Mary attended Royal Palace in Copenhagen, Denmark.",
        "entities": [
            {"text": "Crown Princess Mary", "start": 0, "end": 19},
            {"text": "Royal Palace", "start": 29, "end": 41},
            {"text": "Copenhagen", "start": 45, "end": 55},
            {"text": "Denmark", "start": 57, "end": 64},
        ],
    },
    {
        "text": "Dr. Anthony Fauci Ph.D. directs National Institute of Health.",
        "entities": [
            {"text": "Anthony Fauci", "start": 4, "end": 17},
            {"text": "National Institute of Health", "start": 32, "end": 59},
        ],
    },
    {
        "text": "U.S.A. and U.N. headquarters located in New York.",
        "entities": [
            {"text": "U.S.A.", "start": 0, "end": 6},
            {"text": "U.N.", "start": 11, "end": 15},
            {"text": "New York", "start": 39, "end": 47},
        ],
    },
    {
        "text": "eBay and YouTube founded in San Jose, California.",
        "entities": [
            {"text": "eBay", "start": 0, "end": 4},
            {"text": "YouTube", "start": 9, "end": 16},
            {"text": "San Jose", "start": 27, "end": 35},
            {"text": "California", "start": 37, "end": 47},
        ],
    },
    {
        "text": "iPhone 14 Pro Max released by Apple in Cupertino offices.",
        "entities": [
            {"text": "iPhone 14 Pro Max", "start": 0, "end": 16},
            {"text": "Apple", "start": 27, "end": 32},
            {"text": "Cupertino", "start": 36, "end": 45},
        ],
    },
]

# =============================================================================
# Test Data (30 held-out examples for evaluation)
# =============================================================================

test_examples = [
    {
        "text": "Facebook CEO Mark Zuckerberg announced new metaverse initiative.",
        "entities": [
            {"text": "Facebook", "start": 0, "end": 8},
            {"text": "Mark Zuckerberg", "start": 13, "end": 28},
        ],
    },
    {
        "text": "Elon Musk announced new features for Tesla autonomous driving.",
        "entities": [
            {"text": "Elon Musk", "start": 0, "end": 9},
            {"text": "Tesla", "start": 36, "end": 41},
        ],
    },
    {
        "text": "Google announced Pixel 7 smartphone at Mountain View headquarters.",
        "entities": [
            {"text": "Google", "start": 0, "end": 6},
            {"text": "Pixel 7", "start": 19, "end": 26},
            {"text": "Mountain View", "start": 42, "end": 55},
        ],
    },
    {
        "text": "Samsung Electronics announced new Galaxy S23 in Seoul, South Korea.",
        "entities": [
            {"text": "Samsung Electronics", "start": 0, "end": 19},
            {"text": "Galaxy S23", "start": 33, "end": 43},
            {"text": "Seoul", "start": 47, "end": 52},
            {"text": "South Korea", "start": 54, "end": 65},
        ],
    },
    {
        "text": "Boston Consulting Group opened new office in Berlin, Germany.",
        "entities": [
            {"text": "Boston Consulting Group", "start": 0, "end": 22},
            {"text": "Berlin", "start": 44, "end": 50},
            {"text": "Germany", "start": 52, "end": 59},
        ],
    },
    {
        "text": "McKinsey & Company advises Fortune 500 companies globally.",
        "entities": [
            {"text": "McKinsey & Company", "start": 0, "end": 18},
            {"text": "Fortune 500", "start": 28, "end": 39},
        ],
    },
    {
        "text": "Tim Cook leads Apple with innovation and leadership excellence.",
        "entities": [
            {"text": "Tim Cook", "start": 0, "end": 8},
            {"text": "Apple", "start": 15, "end": 20},
        ],
    },
    {
        "text": "MIT researchers developed breakthrough quantum computing technology.",
        "entities": [
            {"text": "MIT", "start": 0, "end": 3},
        ],
    },
    {
        "text": "Oxford University Press published groundbreaking research paper.",
        "entities": [
            {"text": "Oxford University Press", "start": 0, "end": 22},
        ],
    },
    {
        "text": "Royal Bank of Canada announced quarterly earnings report.",
        "entities": [
            {"text": "Royal Bank of Canada", "start": 0, "end": 19},
        ],
    },
    {
        "text": "Accenture Consulting Group provides IT services worldwide.",
        "entities": [
            {"text": "Accenture Consulting Group", "start": 0, "end": 26},
        ],
    },
    {
        "text": "PwC auditors reviewed financial statements for major corporation.",
        "entities": [
            {"text": "PwC", "start": 0, "end": 3},
        ],
    },
    {
        "text": "Deloitte LLP announced expansion plans for offices in Chicago.",
        "entities": [
            {"text": "Deloitte LLP", "start": 0, "end": 12},
            {"text": "Chicago", "start": 51, "end": 58},
        ],
    },
    {
        "text": "Ernst & Young provided consulting services to Fortune 1000 companies.",
        "entities": [
            {"text": "Ernst & Young", "start": 0, "end": 13},
            {"text": "Fortune 1000", "start": 42, "end": 54},
        ],
    },
    {
        "text": "Airbnb and Uber revolutionized sharing economy in San Francisco.",
        "entities": [
            {"text": "Airbnb", "start": 0, "end": 6},
            {"text": "Uber", "start": 11, "end": 15},
            {"text": "San Francisco", "start": 50, "end": 63},
        ],
    },
    {
        "text": "Spotify Music launched streaming service in Stockholm, Sweden.",
        "entities": [
            {"text": "Spotify Music", "start": 0, "end": 13},
            {"text": "Stockholm", "start": 46, "end": 55},
            {"text": "Sweden", "start": 57, "end": 63},
        ],
    },
    {
        "text": "Slack Technologies acquired by Salesforce for enterprise communication.",
        "entities": [
            {"text": "Slack Technologies", "start": 0, "end": 18},
            {"text": "Salesforce", "start": 30, "end": 40},
        ],
    },
    {
        "text": "Zoom Video Communications expanded during global pandemic.",
        "entities": [
            {"text": "Zoom Video Communications", "start": 0, "end": 25},
        ],
    },
    {
        "text": "DocuSign digitized electronic signature workflows worldwide.",
        "entities": [
            {"text": "DocuSign", "start": 0, "end": 8},
        ],
    },
    {
        "text": "Stripe payments platform founded in Dublin, Ireland.",
        "entities": [
            {"text": "Stripe", "start": 0, "end": 6},
            {"text": "Dublin", "start": 31, "end": 37},
            {"text": "Ireland", "start": 39, "end": 46},
        ],
    },
    {
        "text": "Square Inc. provides payment solutions for small businesses.",
        "entities": [
            {"text": "Square Inc.", "start": 0, "end": 11},
        ],
    },
    {
        "text": "PayPal Holdings processes digital payments globally.",
        "entities": [
            {"text": "PayPal Holdings", "start": 0, "end": 15},
        ],
    },
    {
        "text": "Visa International and Mastercard dominate payment networks.",
        "entities": [
            {"text": "Visa International", "start": 0, "end": 18},
            {"text": "Mastercard", "start": 23, "end": 33},
        ],
    },
    {
        "text": "The Economist magazine published analysis of market trends.",
        "entities": [
            {"text": "Economist", "start": 4, "end": 13},
        ],
    },
    {
        "text": "Bloomberg Businessweek covered startup ecosystem growth.",
        "entities": [
            {"text": "Bloomberg Businessweek", "start": 0, "end": 21},
        ],
    },
    {
        "text": "Fortune magazine ranked Apple as most valuable company.",
        "entities": [
            {"text": "Fortune", "start": 0, "end": 7},
            {"text": "Apple", "start": 31, "end": 36},
        ],
    },
    {
        "text": "Forbes announced billionaire list featuring Elon Musk.",
        "entities": [
            {"text": "Forbes", "start": 0, "end": 6},
            {"text": "Elon Musk", "start": 42, "end": 51},
        ],
    },
    {
        "text": "Reuters reported breaking news from Tokyo, Japan.",
        "entities": [
            {"text": "Reuters", "start": 0, "end": 7},
            {"text": "Tokyo", "start": 31, "end": 36},
            {"text": "Japan", "start": 38, "end": 43},
        ],
    },
    {
        "text": "Associated Press coverage of events in Washington D.C.",
        "entities": [
            {"text": "Associated Press", "start": 0, "end": 16},
            {"text": "Washington D.C.", "start": 38, "end": 53},
        ],
    },
]

print("=" * 80)
print("COMPREHENSIVE NER SPAN EXTRACTION TEST")
print("=" * 80)

print("\n📊 Dataset Summary")
print(f"  Training examples: {len(training_examples)}")
print(f"  Test examples: {len(test_examples)}")
print(
    f"  Total spans in training: {sum(len(e['entities']) for e in training_examples)}"
)
print(f"  Total spans in test: {sum(len(e['entities']) for e in test_examples)}")

# =============================================================================
# Add Training Examples
# =============================================================================

print(f"\n📥 Adding {len(training_examples)} training examples...")
for i, example in enumerate(training_examples, 1):
    text = example["text"]
    entities = example["entities"]

    # Convert to span format (just text positions, no types)
    spans = [
        {"text": e["text"], "start": e["start"], "end": e["end"]} for e in entities
    ]

    chef.add_example(input_data={"text": text}, output_data={"spans": spans})

    if i % 20 == 0:
        print(f"  [{i}/{len(training_examples)}] Added {len(entities)} spans")

print("✓ Added all training examples")
buffer_stats = chef.buffer.get_stats()
print(
    f"  Buffer: {buffer_stats['total_examples']} examples, {buffer_stats['new_corrections']} new corrections"
)

# =============================================================================
# Learn Rules
# =============================================================================

if api_key:
    print(f"\n🤖 Learning rules from {len(training_examples)} examples...")
    chef.learn_rules(run_evaluation=False, max_refinement_iterations=2)
else:
    print("\n⚠️  Skipping rule learning (no API key)")
    print("  Set OPENAI_API_KEY to run with OpenAI")

# =============================================================================
# Evaluate on Test Set
# =============================================================================

print(f"\n📋 Evaluating on {len(test_examples)} test examples...")

all_metrics = {
    "exact_matches": 0,
    "partial_matches": 0,
    "true_positives": 0,
    "false_positives": 0,
    "false_negatives": 0,
    "total_predictions": 0,
    "total_gold": 0,
}

test_results = []

for i, example in enumerate(test_examples, 1):
    text = example["text"]
    gold_entities = example["entities"]
    gold_spans = [
        {"text": e["text"], "start": e["start"], "end": e["end"]} for e in gold_entities
    ]

    # Extract using learned rules
    result = chef.extract({"text": text})
    predicted_spans = result.get("spans", [])

    # Evaluate
    metrics = evaluate_spans(
        predicted_spans, gold_spans, exact_match_only=False, iou_threshold=0.5
    )

    test_results.append(
        {
            "text": text,
            "gold": gold_spans,
            "predicted": predicted_spans,
            "metrics": metrics,
        }
    )

    # Aggregate
    all_metrics["exact_matches"] += metrics["exact_matches"]
    all_metrics["partial_matches"] += metrics["partial_matches"]
    all_metrics["true_positives"] += metrics["true_positives"]
    all_metrics["false_positives"] += metrics["false_positives"]
    all_metrics["false_negatives"] += metrics["false_negatives"]
    all_metrics["total_predictions"] += len(predicted_spans)
    all_metrics["total_gold"] += len(gold_spans)

    if i % 10 == 0:
        print(f"  [{i}/{len(test_examples)}] Evaluated")

# =============================================================================
# Generate Report
# =============================================================================

print("\n" + "=" * 80)
print("EVALUATION RESULTS")
print("=" * 80)

tp = all_metrics["true_positives"]
fp = all_metrics["false_positives"]
fn = all_metrics["false_negatives"]

precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
f1 = (
    2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
)

exact_accuracy = (
    all_metrics["exact_matches"] / all_metrics["total_gold"]
    if all_metrics["total_gold"] > 0
    else 0.0
)
partial_accuracy = (
    tp / all_metrics["total_gold"] if all_metrics["total_gold"] > 0 else 0.0
)

print("\n📊 SPAN EXTRACTION METRICS")
print(f"{'─' * 80}")
print(
    f"Exact Match Accuracy:     {exact_accuracy:>6.1%}  ({all_metrics['exact_matches']}/{all_metrics['total_gold']} spans)"
)
print(
    f"Partial Match Accuracy:   {partial_accuracy:>6.1%}  ({tp}/{all_metrics['total_gold']} spans, IoU > 0.5)"
)
print()
print(f"Precision:  {precision:>6.1%}  (TP / (TP + FP))")
print(f"Recall:     {recall:>6.1%}  (TP / (TP + FN))")
print(f"F1 Score:   {f1:>6.1%}  (Harmonic mean)")
print()
print(f"True Positives:   {tp}")
print(f"False Positives:  {fp}")
print(f"False Negatives:  {fn}")
print()

# Show learned rules if available
if chef.dataset.rules:
    print(f"✓ Learned {len(chef.dataset.rules)} rules:")
    for rule in chef.dataset.rules[:5]:
        print(f"  - {rule.name}: {rule.content[:60]}...")
    if len(chef.dataset.rules) > 5:
        print(f"  ... and {len(chef.dataset.rules) - 5} more rules")
else:
    print("⚠️  No rules learned (need API key for LLM)")

# Show some example results
print("\n📝 EXAMPLE RESULTS")
print(f"{'─' * 80}")

success_count = 0
failure_count = 0

for result in test_results:
    if (
        result["metrics"]["false_positives"] == 0
        and result["metrics"]["false_negatives"] == 0
    ):
        success_count += 1
    else:
        failure_count += 1

print(f"Fully correct examples: {success_count}/{len(test_results)}")
print(f"Examples with errors:  {failure_count}/{len(test_results)}")

# Show first failure
if failure_count > 0:
    print("\nFirst failure example:")
    for result in test_results:
        if (
            result["metrics"]["false_positives"] > 0
            or result["metrics"]["false_negatives"] > 0
        ):
            print(f"  Text: {result['text'][:70]}...")
            print(f"  Gold:      {[s['text'] for s in result['gold']]}")
            print(f"  Predicted: {[s['text'] for s in result['predicted']]}")
            break

print(f"\n{'=' * 80}")
print("✓ Test complete!")
print(f"{'=' * 80}\n")
