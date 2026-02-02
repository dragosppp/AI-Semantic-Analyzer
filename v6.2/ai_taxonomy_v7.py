"""
═══════════════════════════════════════════════════════════════════════════════
AI SEMANTIC ANALYZER v7.0 - DUAL TAXONOMY KEYWORDS (REFINED)
═══════════════════════════════════════════════════════════════════════════════

RESTRUCTURARE: 
    - Dimension 1: AI Applications (8 categories) - PENTRU CE?
    - Dimension 2: AI Technologies (8 categories) - CE TEHNOLOGIE?

IMPROVEMENTS v7.0.1:
    - Removed risky standalone keywords (research, lab, digitalization, ML)
    - Enhanced patterns with mandatory AI context
    - Added keyword_tiers metadata (1=high, 2=medium, 3=low confidence)
    - Added detection_method tracking for pattern vs keyword hits
    - Clarified B6 scope (agentic AI = autonomous workflows, not generic agents)

Autor: TeRa0
Versiune: 7.0.1
Data: Februarie 2026
═══════════════════════════════════════════════════════════════════════════════
"""

# ═══════════════════════════════════════════════════════════════════════════
# DIMENSION 1: AI APPLICATIONS & USE CASES
# ═══════════════════════════════════════════════════════════════════════════

AI_APPLICATIONS = {
    # ═══════════════════════════════════════════════════════════════════
    # FUNCTIONAL / OPERATIONAL APPLICATIONS
    # ═══════════════════════════════════════════════════════════════════
    
    'A1_Product_Innovation': {
        'name': 'Product & Service Innovation',
        'description': 'AI-enhanced products and services; New product features; R&D acceleration',
        'keywords': [
            # Product development - HIGH CONFIDENCE (Tier 1)
            'AI-powered product', 'AI-enabled', 'intelligent product', 'smart product',
            'AI feature', 'AI-first', 'next-generation product', 'intelligent system',
            
            # R&D & Innovation - SELECTIVE (removed standalone: research, lab, innovation, development)
            'R&D', 'prototype', 'design', 'patent', 'breakthrough', 'discovery',
            'innovation center', 'frontier research',
            
            # Collaboration - SAFE
            'academic collaboration', 'university partnership', 'research center',
            'AI research', 'ML research',
        ],
        'patterns': [
            r'AI[\s-]?powered.*(?:product|feature|solution)',
            r'intelligent.*(?:system|solution|product)',
            r'R&D.*(?:AI|artificial intelligence|machine learning)',
            r'(?:develop|build|create).*AI.*(?:product|solution)',
            r'(?:AI|ML)\s+(?:patent|research|lab|center)',
            r'(?:research|innovation|lab).*(?:AI|artificial intelligence|machine learning)',  # AI context mandatory
            r'(?:AI|ML)\s+(?:breakthrough|innovation|discovery)',
            r'(?:academic|university).*(?:AI|ML)\s+(?:research|collaboration)',
        ],
        'keyword_tiers': {
            # Tier 1: High confidence
            'AI-powered product': 1, 'AI-enabled': 1, 'intelligent product': 1,
            'smart product': 1, 'AI feature': 1, 'AI-first': 1,
            'AI research': 1, 'ML research': 1,
            # Tier 2: Medium confidence
            'prototype': 2, 'design': 2, 'patent': 2,
            'breakthrough': 2, 'discovery': 2,
            # Tier 3: Low confidence (supportive)
            'R&D': 3, 'innovation center': 3, 'frontier research': 3,
        }
    },
    
    'A2_Operational_Excellence': {
        'name': 'Operational Excellence & Automation',
        'description': 'Process automation; Supply chain optimization; Operational efficiency',
        'keywords': [
            # Automation - HIGH CONFIDENCE (Tier 1)
            'deploy', 'automate', 'automation', 'RPA', 'intelligent automation',
            'hyperautomation', 'IPA', 'process automation',
            
            # Operational - SELECTIVE (removed: digitalization, workflow as standalone)
            'efficiency', 'streamline', 'optimize', 
            'process improvement', 'implementation', 'rollout', 'integration',
            'operational excellence', 'productivity',
            
            # Supply chain & Manufacturing - DOMAIN SPECIFIC (Tier 1)
            'supply chain', 'logistics', 'manufacturing', 'production',
            'quality control', 'predictive maintenance', 'resource allocation',
        ],
        'patterns': [
            r'deploy(?:ed|ing|ment).*(?:AI|ML|machine learning)',
            r'automat(?:e|ed|ing|ion).*(?:process|workflow|task)',
            r'(?:AI|ML).*(?:efficiency|optimization|productivity)',
            r'(?:intelligent|hyper)[\s-]?automation',
            r'(?:supply chain|logistics).*(?:AI|ML|optimization)',
            r'(?:predictive|preventive)\s+maintenance',
            r'(?:workflow|digitalization).*(?:AI|ML|automation)',  # AI context mandatory
            r'(?:process|operational).*(?:optimization|excellence).*(?:AI|ML)',
        ],
        'keyword_tiers': {
            # Tier 1: High confidence
            'intelligent automation': 1, 'hyperautomation': 1, 'RPA': 1,
            'process automation': 1, 'predictive maintenance': 1,
            'supply chain': 1, 'logistics': 1,
            # Tier 2: Medium confidence
            'efficiency': 2, 'optimize': 2, 'streamline': 2,
            'productivity': 2, 'operational excellence': 2,
            # Tier 3: Low confidence (supportive)
            'deploy': 3, 'implementation': 3, 'rollout': 3, 'integration': 3,
        }
    },
    
    'A3_Customer_Experience': {
        'name': 'Customer Experience & Engagement',
        'description': 'Customer service automation; Personalization; Marketing optimization',
        'keywords': [
            # Customer service
            'chatbot', 'virtual assistant', 'conversational AI', 'customer service',
            'support', 'customer experience', 'CX', 'customer journey',
            
            # Personalization
            'personalization', 'recommendation', 'recommender system',
            'engagement', 'targeted', 'customized',
            
            # Marketing & Sales
            'marketing automation', 'sales optimization', 'lead scoring',
            'customer insights', 'sentiment analysis', 'voice of customer',
        ],
        'patterns': [
            r'(?:AI|ML).*(?:chatbot|virtual assistant|conversational)',
            r'personali[zs](?:e|ed|ation).*(?:customer|experience)',
            r'customer.*(?:experience|service).*(?:AI|ML)',
            r'(?:recommendation|recommender)\s+(?:system|engine)',
            r'(?:AI|ML).*(?:sentiment|customer\s+insights)',
        ]
    },
    
    'A4_Risk_Compliance': {
        'name': 'Risk Management & Compliance',
        'description': 'Fraud detection; Cybersecurity; Regulatory compliance; Risk assessment',
        'keywords': [
            # Fraud & Security
            'fraud detection', 'fraud prevention', 'cybersecurity', 'threat detection',
            'security', 'anomaly detection', 'intrusion detection',
            
            # Compliance
            'compliance', 'regulation', 'regulatory', 'AML', 'KYC',
            'anti-money laundering', 'GDPR', 'audit', 'monitoring',
            
            # Risk management
            'risk management', 'risk assessment', 'credit risk', 'model risk',
            'risk scoring', 'underwriting',
        ],
        'patterns': [
            r'(?:fraud|anomaly)\s+detection',
            r'(?:cyber|threat)\s+(?:security|detection)',
            r'(?:AML|KYC|anti[\s-]?money\s+laundering)',
            r'(?:compliance|regulatory).*(?:AI|ML|automation)',
            r'(?:risk|credit)\s+(?:assessment|scoring|management)',
        ]
    },
    
    'A5_Data_Analytics': {
        'name': 'Data Analytics & Business Intelligence',
        'description': 'Predictive analytics; Business intelligence; Data-driven decision making',
        'keywords': [
            # Analytics
            'predictive analytics', 'forecasting', 'prediction', 'insights',
            'business intelligence', 'BI', 'analytics', 'advanced analytics',
            
            # Data-driven
            'data-driven', 'data science', 'modeling', 'algorithm',
            'pattern recognition', 'trend analysis',
            
            # Decision support
            'decision support', 'optimization', 'performance analytics',
        ],
        'patterns': [
            r'predictive\s+(?:analytics|model|insight)',
            r'machine learning.*(?:model|algorithm|prediction)',
            r'data[\s-]?driven.*(?:insight|decision|strategy)',
            r'(?:advanced|augmented)\s+analytics',
            r'business\s+intelligence.*(?:AI|ML)',
        ]
    },
    
    # ═══════════════════════════════════════════════════════════════════
    # STRATEGIC / ORGANIZATIONAL APPLICATIONS
    # ═══════════════════════════════════════════════════════════════════
    
    'A6_Strategy_Investment': {
        'name': 'AI Strategy & Investment',
        'description': 'Strategic initiatives; Budget allocation; Partnerships; ROI measurement',
        'keywords': [
            # Investment & Budget
            'invest', 'investment', 'acquisition', 'M&A', 'budget', 'funding',
            'capital', 'allocated', 'committed funds', 'AI spending', 'AI budget',
            
            # Strategy
            'strategic initiative', 'AI strategy', 'transformation', 'roadmap',
            'AI transformation', 'digital transformation', 'strategic priority',
            
            # Partnerships
            'partnership', 'venture', 'collaboration', 'ecosystem',
            'acquisition', 'pilot project', 'proof of concept', 'POC',
            
            # Centers & Programs
            'AI center', 'center of excellence', 'innovation lab', 'AI program',
        ],
        'patterns': [
            r'invest(?:ed|ing|ment).*(?:AI|artificial intelligence)',
            r'(?:AI|artificial intelligence).*(?:budget|funding|capital)',
            r'acquir(?:ed|ing|ition).*(?:AI|machine learning)',
            r'\$[\d,.]+\s*(?:million|billion|M|B).*(?:AI|artificial intelligence)',
            r'(?:AI|ML)\s+(?:strategy|transformation|initiative)',
            r'(?:center\s+of\s+excellence|innovation\s+lab).*AI',
        ]
    },
    
    'A7_Governance_Ethics': {
        'name': 'AI Governance & Ethics',
        'description': 'Responsible AI; Bias mitigation; Explainability; Ethical guidelines',
        'keywords': [
            # Responsible AI
            'responsible AI', 'ethical AI', 'trustworthy AI', 'AI ethics',
            'AI governance', 'AI policy', 'governance framework',
            
            # Fairness & Bias
            'bias', 'fairness', 'bias detection', 'bias mitigation',
            'discrimination', 'equity', 'inclusive AI',
            
            # Transparency & Explainability
            'explainability', 'XAI', 'interpretability', 'transparency',
            'accountability', 'model interpretability',
            
            # Safety & Risk
            'AI safety', 'guardrails', 'hallucination', 'alignment',
            'red teaming', 'model evaluation', 'risk framework',
        ],
        'patterns': [
            r'responsible\s+(?:AI|artificial intelligence)',
            r'(?:AI|ML).*(?:ethics|governance|compliance)',
            r'(?:bias|fairness).*(?:detection|mitigation|audit)',
            r'AI\s+(?:risk|policy|governance)\s+framework',
            r'(?:explainab|interpretab)(?:le|ility)',
            r'(?:trustworthy|transparent|accountable)\s+(?:AI|ML)',
        ]
    },
    
    'A8_Talent_Workforce': {
        'name': 'AI Talent & Workforce Development',
        'description': 'Training; Upskilling; Talent acquisition; Change management',
        'keywords': [
            # Skills & Training
            'upskill', 'reskill', 'training', 'education', 'learning',
            'AI literacy', 'capability building', 'skill development',
            
            # Talent
            'AI talent', 'hiring', 'recruitment', 'data scientist',
            'ML engineer', 'AI team', 'AI expertise', 'talent acquisition',
            
            # Organizational
            'workforce', 'organizational change', 'change management',
            'AI adoption', 'cultural transformation', 'employee training',
        ],
        'patterns': [
            r'(?:AI|ML).*(?:talent|skill|training|team)',
            r'(?:upskill|reskill)(?:ing)?.*(?:AI|data|digital)',
            r'hir(?:e|ed|ing).*(?:AI|ML|data).*(?:engineer|scientist)',
            r'(?:workforce|employee).*(?:training|development).*AI',
            r'AI\s+(?:literacy|capability|adoption)\s+program',
        ]
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# DIMENSION 2: AI TECHNOLOGIES & TECHNICAL CAPABILITIES
# ═══════════════════════════════════════════════════════════════════════════

AI_TECHNOLOGIES = {
    'B1_Traditional_ML': {
        'name': 'Traditional Machine Learning',
        'description': 'Supervised/unsupervised learning; Ensemble methods; Statistical models',
        'keywords': [
            # Core ML - NO standalone "ML" (too ambiguous with "ML of water")
            'machine learning',  # Full phrase only
            'supervised learning', 'unsupervised learning',
            'classification', 'regression', 'clustering',
            
            # Algorithms - HIGH CONFIDENCE (Tier 1)
            'random forest', 'decision tree', 'logistic regression',
            'support vector machine', 'SVM', 'k-means', 'XGBoost',
            'gradient boosting', 'ensemble', 'bagging', 'boosting',
            
            # Process - MEDIUM CONFIDENCE (Tier 2)
            'feature engineering', 'model training', 'cross-validation',
            'hyperparameter tuning', 'overfitting', 'underfitting',
        ],
        'patterns': [
            r'machine\s+learning(?!\s+(?:operation|platform|infrastructure))',
            r'\bML\b(?!\s*(?:Ops|infrastructure|platform|of|metric|tons?|liters?|gallons?|\d))',  # Enhanced exclusions
            r'(?:supervised|unsupervised)\s+learning',
            r'(?:random\s+forest|decision\s+tree|XGBoost)',
            r'(?:classification|regression|clustering)\s+(?:model|algorithm)',
            r'ML\s+(?:model|algorithm|technique|approach|system)',  # ML only with technical context
            r'(?:train|build|develop).*ML.*(?:model|algorithm)',
        ],
        'keyword_tiers': {
            # Tier 1: High confidence
            'machine learning': 1, 'supervised learning': 1, 'unsupervised learning': 1,
            'random forest': 1, 'decision tree': 1, 'XGBoost': 1,
            'gradient boosting': 1, 'support vector machine': 1, 'SVM': 1,
            # Tier 2: Medium confidence
            'classification': 2, 'regression': 2, 'clustering': 2,
            'ensemble': 2, 'bagging': 2, 'boosting': 2,
            # Tier 3: Low confidence (supportive)
            'feature engineering': 3, 'model training': 3, 'cross-validation': 3,
            'hyperparameter tuning': 3, 'overfitting': 3, 'underfitting': 3,
        }
    },
    
    'B2_Deep_Learning': {
        'name': 'Deep Learning & Neural Networks',
        'description': 'DNN, CNN, RNN; Reinforcement learning; Transfer learning',
        'keywords': [
            # Core DL
            'deep learning', 'DL', 'neural network', 'artificial neural network',
            'deep neural network', 'DNN',
            
            # Architectures
            'convolutional neural network', 'CNN', 'convolution',
            'recurrent neural network', 'RNN', 'LSTM', 'GRU',
            'autoencoder', 'GAN', 'generative adversarial',
            
            # Techniques
            'backpropagation', 'gradient descent', 'activation function',
            'dropout', 'batch normalization', 'transfer learning',
            'reinforcement learning', 'Q-learning', 'policy gradient',
        ],
        'patterns': [
            r'deep\s+learning',
            r'\bDL\b(?!\s+(?:of|metric))',  # Exclude "DL of water"
            r'(?:deep\s+)?neural\s+network',
            r'\b(?:CNN|RNN|LSTM|GRU)\b',
            r'(?:convolutional|recurrent)\s+neural',
            r'reinforcement\s+learning',
        ]
    },
    
    'B3_NLP_NonLLM': {
        'name': 'Natural Language Processing (Non-LLM)',
        'description': 'Text classification; NER; Traditional NLP; Sentiment analysis',
        'keywords': [
            # Core NLP
            'natural language processing', 'NLP', 'text analysis',
            'text mining', 'text classification', 'information extraction',
            
            # Techniques (non-LLM)
            'named entity recognition', 'NER', 'part-of-speech', 'POS tagging',
            'dependency parsing', 'tokenization', 'lemmatization',
            'sentiment analysis', 'opinion mining',
            
            # Traditional embeddings
            'word embedding', 'Word2Vec', 'GloVe', 'fastText',
            'topic modeling', 'TF-IDF',
        ],
        'patterns': [
            r'natural\s+language\s+processing(?!\s+(?:model|LLM))',
            r'\bNLP\b(?!\s+(?:model|LLM))',
            r'(?:text|sentiment)\s+(?:analysis|classification)',
            r'named\s+entity\s+recognition',
            r'(?:word|text)\s+embedding(?!.*(?:LLM|large\s+language))',
        ]
    },
    
    'B4_GenAI_LLMs': {
        'name': 'Generative AI & Large Language Models',
        'description': 'ChatGPT, Claude, Gemini; Text/code generation; Foundation models',
        'keywords': [
            # GenAI General
            'generative AI', 'GenAI', 'generative model', 'generation',
            
            # LLMs
            'large language model', 'LLM', 'foundation model',
            'transformer', 'attention mechanism', 'self-attention',
            
            # Specific models
            'GPT', 'ChatGPT', 'GPT-3', 'GPT-4', 'GPT-4o',
            'Claude', 'Anthropic', 'Gemini', 'Bard', 'PaLM',
            'Llama', 'LLaMA', 'Mistral', 'Cohere',
            
            # Tools & Assistants
            'Copilot', 'GitHub Copilot', 'Microsoft 365 Copilot',
            
            # Techniques
            'prompt engineering', 'few-shot', 'zero-shot', 'chain-of-thought',
            'fine-tuning', 'RLHF', 'instruction tuning',
            'RAG', 'retrieval-augmented', 'vector database', 'embedding store',
            
            # Applications
            'text generation', 'content generation', 'code generation',
            'text-to-image', 'image generation', 'DALL-E', 'Stable Diffusion', 'Midjourney',
            'multimodal', 'vision-language',
        ],
        'patterns': [
            r'generativ(?:e)?\s*(?:AI|artificial intelligence)',
            r'(?:large\s+)?language\s+model(?:s)?',
            r'\bLLM(?:s)?\b',
            r'\bGPT[\s-]?[3-5]?(?:o)?\b',
            r'\bChatGPT\b',
            r'\b(?:Claude|Gemini|Bard|PaLM)\b(?!\s+(?:project|program))',  # Exclude "Gemini project"
            r'foundation\s+model(?:s)?',
            r'\bGenAI\b',
            r'(?:retrieval[\s-]?augmented|RAG)',
            r'(?:vector|embedding)\s+(?:database|store)',
            r'(?:prompt|instruction)\s+(?:engineering|tuning)',
            r'(?:text|image|code)\s+generation',
        ]
    },
    
    'B5_Computer_Vision': {
        'name': 'Computer Vision',
        'description': 'Image recognition; Object detection; Video analytics; Visual inspection',
        'keywords': [
            # Core CV
            'computer vision', 'CV', 'image recognition', 'image processing',
            'visual recognition', 'image analysis',
            
            # Techniques
            'object detection', 'object recognition', 'image classification',
            'semantic segmentation', 'instance segmentation',
            'facial recognition', 'face detection', 'face recognition',
            'OCR', 'optical character recognition',
            
            # Applications
            'video analytics', 'video analysis', 'visual inspection',
            'quality inspection', 'defect detection', 'visual quality control',
            'image segmentation', '3D vision', 'depth estimation',
            
            # Architectures
            'ResNet', 'VGG', 'YOLO', 'R-CNN', 'EfficientNet',
        ],
        'patterns': [
            r'computer\s+vision',
            r'(?:image|object|face|facial)\s+(?:recognition|detection|classification)',
            r'visual\s+(?:recognition|inspection|analysis)',
            r'video\s+analytics',
            r'(?:OCR|optical\s+character\s+recognition)',
            r'\b(?:YOLO|ResNet|VGG|R-CNN)\b',
        ]
    },
    
    'B6_Robotics_Autonomous': {
        'name': 'Robotics & Autonomous Systems',
        'description': 'Autonomous vehicles; AI-powered robotics; Agentic AI workflows (not generic copilot agents); Multi-agent systems',
        'keywords': [
            # Autonomous systems - HIGH CONFIDENCE (Tier 1)
            'autonomous', 'self-driving', 'autonomous vehicle',
            'autonomous navigation', 'path planning',
            
            # Robotics (AI-powered) - HIGH CONFIDENCE (Tier 1)
            'intelligent robot', 'cognitive robot', 'autonomous robot',
            'collaborative robot', 'cobot', 'robot learning',
            
            # Agentic AI - SPECIFIC to autonomous workflows (removed generic: agent, workflow agent)
            'agentic', 'AI agent', 'autonomous agent',
            'multi-agent', 'agent orchestration', 'agent framework',
            'tool use', 'function calling', 'computer use',
            
            # Specific systems - HIGH CONFIDENCE (Tier 1)
            'drone', 'UAV', 'unmanned aerial', 'AGV', 'automated guided vehicle',
        ],
        'patterns': [
            r'(?:autonomous|self[\s-]?driving)\s+(?:vehicle|system|car)',
            r'(?:agentic)\s+(?:system|AI|workflow)',  # "agentic" is specific enough
            r'(?:autonomous\s+)?AI\s+agent(?:s)?',  # Needs "autonomous" or "AI" prefix
            r'multi[\s-]?agent\s+(?:system|orchestration)',
            r'agent\s+(?:orchestration|framework|coordination)',  # Needs technical context
            r'(?:tool|function)\s+(?:use|calling)',
            r'(?:drone|UAV|AGV).*(?:AI|autonomous|intelligent)',
            r'(?:cognitive|intelligent|autonomous)\s+robot',
            r'(?:collaborative\s+)?robot.*(?:AI|learning|intelligent)',
        ],
        'keyword_tiers': {
            # Tier 1: High confidence
            'autonomous': 1, 'self-driving': 1, 'autonomous vehicle': 1,
            'intelligent robot': 1, 'cognitive robot': 1, 'autonomous robot': 1,
            'agentic': 1, 'multi-agent': 1, 'drone': 1, 'UAV': 1, 'AGV': 1,
            # Tier 2: Medium confidence
            'AI agent': 2, 'autonomous agent': 2, 'cobot': 2,
            'agent orchestration': 2, 'agent framework': 2,
            # Tier 3: Low confidence (supportive)
            'tool use': 3, 'function calling': 3, 'computer use': 3,
            'path planning': 3, 'robot learning': 3,
        }
    },
    
    'B7_Infrastructure_Platforms': {
        'name': 'AI Infrastructure & Platforms',
        'description': 'Cloud AI; MLOps; GPU infrastructure; Model deployment; Enterprise platforms',
        'keywords': [
            # MLOps
            'MLOps', 'ML operations', 'model ops', 'AI ops', 'AIOps',
            'CI/CD', 'continuous integration', 'model deployment',
            'model serving', 'model hosting', 'model monitoring',
            
            # Cloud & Infrastructure
            'cloud AI', 'AI platform', 'AI infrastructure', 'ML platform',
            'AI-as-a-service', 'AIaaS', 'ML-as-a-service',
            
            # Cloud providers
            'AWS', 'SageMaker', 'Amazon Bedrock',
            'Azure AI', 'Azure ML', 'Azure OpenAI',
            'Google Cloud AI', 'Vertex AI', 'Google AI Platform',
            
            # Hardware
            'GPU', 'TPU', 'NPU', 'AI accelerator', 'GPU cluster',
            'NVIDIA', 'CUDA', 'inference engine',
            
            # Edge & Compute
            'edge AI', 'edge computing', 'edge deployment',
            'compute', 'scalability', 'inference',
            
            # Enterprise platforms
            'IBM Watson', 'Watsonx', 'AI platform',
            
            # Development tools
            'Kubernetes', 'Docker', 'MLflow', 'Kubeflow',
            'model registry', 'feature store',
        ],
        'patterns': [
            r'\bMLOps\b',
            r'(?:AI|ML)\s+(?:platform|infrastructure|stack)',
            r'(?:edge|cloud)\s+(?:AI|computing|deployment)',
            r'(?:model|AI)\s+(?:deployment|serving|hosting|monitoring)',
            r'(?:GPU|TPU|NPU)\s+(?:compute|cluster|infrastructure)',
            r'AI[\s-]?as[\s-]?a[\s-]?service',
            r'(?:AWS|Azure|Google\s+Cloud).*AI',
            r'(?:SageMaker|Vertex|Watson)',
        ]
    },
    
    'B8_General_AI': {
        'name': 'AI (General/Unspecified)',
        'description': 'Generic AI references without technical specificity',
        'keywords': [
            # Generic terms
            'artificial intelligence', 'AI', 'AI system', 'AI solution',
            'AI technology', 'AI capability', 'intelligent system',
            'cognitive', 'smart', 'intelligent',
        ],
        'patterns': [
            r'\bartificial\s+intelligence\b',
            r'\bAI\b(?!\s*(?:powered|enabled|first))',  # AI without specific context
            r'AI\s+(?:system|solution|technology|capability)',
        ]
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# MAPPINGS: OLD CATEGORIES → NEW TAXONOMY
# ═══════════════════════════════════════════════════════════════════════════

CATEGORY_MAPPING_V6_TO_V7 = {
    # Old v6.1 → New v7.0 (Applications)
    'Product Development': 'A1_Product_Innovation',
    'Research & Innovation': 'A1_Product_Innovation',
    'Operational Implementation': 'A2_Operational_Excellence',
    'Customer Experience': 'A3_Customer_Experience',
    'Risk & Compliance': 'A4_Risk_Compliance',
    'Data & Analytics': 'A5_Data_Analytics',
    'Strategic Investment': 'A6_Strategy_Investment',
    'Explainability & AI Safety': 'A7_Governance_Ethics',
    'Talent & Workforce': 'A8_Talent_Workforce',
    
    # Old v6.1 → New v7.0 (Technologies)
    'Generative AI & LLMs': 'B4_GenAI_LLMs',
    'AI Infrastructure & MLOps': 'B7_Infrastructure_Platforms',
    'AI Coding & Development': 'B7_Infrastructure_Platforms',
    'Agentic AI Systems': 'B6_Robotics_Autonomous',
}


# ═════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_all_application_keywords():
    """Returns all keywords from all application categories"""
    keywords = []
    for category in AI_APPLICATIONS.values():
        keywords.extend(category['keywords'])
    return list(set(keywords))


def get_all_technology_keywords():
    """Returns all keywords from all technology categories"""
    keywords = []
    for category in AI_TECHNOLOGIES.values():
        keywords.extend(category['keywords'])
    return list(set(keywords))


def get_category_info(category_code: str):
    """Get information about a specific category"""
    # Check applications
    if category_code in AI_APPLICATIONS:
        return AI_APPLICATIONS[category_code]
    
    # Check technologies
    if category_code in AI_TECHNOLOGIES:
        return AI_TECHNOLOGIES[category_code]
    
    return None


def get_keyword_tier(category_code: str, keyword: str) -> int:
    """
    Get confidence tier for a keyword in a category.
    Returns: 1 (high), 2 (medium), 3 (low), or 2 (default if not specified)
    """
    category = get_category_info(category_code)
    if not category:
        return 2  # Default medium
    
    tiers = category.get('keyword_tiers', {})
    return tiers.get(keyword, 2)  # Default to tier 2 if not specified


def get_high_confidence_keywords(category_code: str) -> list:
    """Get only Tier 1 (high confidence) keywords for a category"""
    category = get_category_info(category_code)
    if not category:
        return []
    
    tiers = category.get('keyword_tiers', {})
    return [kw for kw, tier in tiers.items() if tier == 1]


def print_taxonomy_summary():
    """Print a summary of the dual taxonomy"""
    print("="*80)
    print("DUAL TAXONOMY v7.0.1 - SUMMARY WITH KEYWORD TIERS")
    print("="*80)
    
    print("\nDIMENSION 1: AI APPLICATIONS (8 categories)")
    print("-" * 80)
    for code, info in AI_APPLICATIONS.items():
        total_kw = len(info['keywords'])
        tiers = info.get('keyword_tiers', {})
        tier1 = len([k for k, t in tiers.items() if t == 1])
        tier2 = len([k for k, t in tiers.items() if t == 2])
        tier3 = len([k for k, t in tiers.items() if t == 3])
        
        print(f"{code}: {info['name']}")
        print(f"   Total: {total_kw} keywords, {len(info['patterns'])} patterns")
        if tiers:
            print(f"   Tiers: {tier1} high, {tier2} medium, {tier3} low")
    
    print("\nDIMENSION 2: AI TECHNOLOGIES (8 categories)")
    print("-" * 80)
    for code, info in AI_TECHNOLOGIES.items():
        total_kw = len(info['keywords'])
        tiers = info.get('keyword_tiers', {})
        tier1 = len([k for k, t in tiers.items() if t == 1])
        tier2 = len([k for k, t in tiers.items() if t == 2])
        tier3 = len([k for k, t in tiers.items() if t == 3])
        
        print(f"{code}: {info['name']}")
        print(f"   Total: {total_kw} keywords, {len(info['patterns'])} patterns")
        if tiers:
            print(f"   Tiers: {tier1} high, {tier2} medium, {tier3} low")
    
    total_app_keywords = len(get_all_application_keywords())
    total_tech_keywords = len(get_all_technology_keywords())
    
    print("\n" + "="*80)
    print(f"Total unique application keywords: {total_app_keywords}")
    print(f"Total unique technology keywords: {total_tech_keywords}")
    print(f"Total categories: 16 (8 applications + 8 technologies)")
    print("\nSAFETY IMPROVEMENTS v7.0.1:")
    print("  - Removed standalone risky keywords: research, lab, digitalization, workflow, ML, agent")
    print("  - Added mandatory AI context in patterns for generic terms")
    print("  - Implemented keyword_tiers (1=high, 2=medium, 3=low confidence)")
    print("  - Enhanced false positive exclusions in B1 (ML measurements)")
    print("  - Clarified B6 scope (agentic = autonomous workflows, not generic copilots)")
    print("="*80)


if __name__ == "__main__":
    print_taxonomy_summary()
