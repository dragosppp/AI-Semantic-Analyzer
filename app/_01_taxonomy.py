"""
═══════════════════════════════════════════════════════════════════════════════
AI SEMANTIC ANALYZER - SEMANTIC TAXONOMIES
═══════════════════════════════════════════════════════════════════════════════

Two semantic taxonomies are classified in a SINGLE run; every detected reference
is labelled under BOTH.

CLASSIC taxonomy (business-oriented):
    - Dimension 1: AI Applications   (7 categories, A1–A7) — "WHAT FOR?"
    - Dimension 2: AI Technologies   (8 categories, B1–B8) — "WHAT TECHNOLOGY?"

EU_SEMANTICS taxonomy (European Commission JRC "AI Watch"):
    - 8 domains / 12 subdomain leaves — "WHICH AI CAPABILITY?"

Each classic category exposes keyword_tiers metadata (1=high, 2=medium,
3=low confidence). Patterns require AI context; B6 scope = autonomous workflows
(not copilots).
═══════════════════════════════════════════════════════════════════════════════
"""

from __future__ import annotations

# ═══════════════════════════════════════════════════════════════════════════
# DIMENSION 1: AI APPLICATIONS & USE CASES
# ═══════════════════════════════════════════════════════════════════════════

AI_APPLICATIONS: dict[str, dict] = {

    'A1_Strategic_Transformation': {
        'name': 'Strategic Decision-Making & Business Model Transformation',
        'description': 'Strategic/executive decision-making; corporate, business & digital '
                       'strategy; business model & organizational transformation; value creation',
        'keywords': [
            # Decision-making
            'strategic decision', 'strategic decisions', 'business decision', 'business decisions',
            'managerial decision', 'managerial decisions', 'executive decision', 'executive decisions',
            'decision making', 'decision-making',
            # Strategy
            'strategic planning', 'corporate strategy', 'business strategy', 'digital strategy',
            'competitive advantage', 'strategic alignment',
            # Business model & transformation
            'business model', 'business models', 'business model innovation',
            'digital transformation', 'organizational transformation', 'enterprise transformation',
            # Value & performance
            'value creation', 'value capture', 'firm performance',
            'market intelligence', 'scenario planning',
        ],
        'patterns': [
            r'(?:AI|ML)\s+(?:strategy|transformation|roadmap)',
            r'digital\s+transformation',
            r'business\s+model\s+innovation',
            r'(?:strategic|executive|managerial)\s+decision[\s-]?making',
        ],
        'keyword_tiers': {
            'digital transformation': 1, 'organizational transformation': 1,
            'enterprise transformation': 1, 'business model innovation': 1,
            'scenario planning': 1, 'corporate strategy': 1, 'digital strategy': 1,
            'strategic planning': 1,
            'strategic decision': 2, 'strategic decisions': 2, 'business decision': 2,
            'business decisions': 2, 'managerial decision': 2, 'managerial decisions': 2,
            'executive decision': 2, 'executive decisions': 2, 'decision making': 2,
            'decision-making': 2, 'business strategy': 2, 'competitive advantage': 2,
            'strategic alignment': 2, 'market intelligence': 2, 'business model': 2,
            'business models': 2,
            'value creation': 3, 'value capture': 3, 'firm performance': 3,
        },
    },

    'A2_Operational_Optimization': {
        'name': 'Operational Process Optimization',
        'description': 'Operational efficiency & excellence; process & workflow automation (RPA); '
                       'supply chain & inventory optimization; predictive maintenance',
        'keywords': [
            # Efficiency & excellence
            'operational efficiency', 'operational excellence', 'process optimization',
            'productivity improvement', 'cost reduction',
            # Process & automation
            'business process', 'business processes', 'process automation', 'workflow automation',
            'intelligent automation', 'robotic process automation', 'RPA', 'process mining',
            # Quality
            'quality control', 'quality improvement',
            # Supply chain & production
            'supply chain optimization', 'inventory optimization', 'predictive maintenance',
            'resource allocation', 'production planning', 'operations management',
            # Generic / supportive
            'scheduling', 'lean',
        ],
        'patterns': [
            r'(?:robotic\s+process\s+automation|intelligent\s+automation|hyperautomation)',
            r'\bRPA\b',
            r'(?:supply\s+chain|inventory)\s+optimi[zs]ation',
            r'process\s+(?:optimi[zs]ation|automation|mining)',
            r'predictive\s+maintenance',
        ],
        'keyword_tiers': {
            'robotic process automation': 1, 'RPA': 1, 'intelligent automation': 1,
            'process automation': 1, 'workflow automation': 1, 'predictive maintenance': 1,
            'supply chain optimization': 1, 'inventory optimization': 1, 'process mining': 1,
            'operational excellence': 1,
            'operational efficiency': 2, 'process optimization': 2, 'business process': 2,
            'business processes': 2, 'productivity improvement': 2, 'cost reduction': 2,
            'quality control': 2, 'quality improvement': 2, 'resource allocation': 2,
            'production planning': 2, 'operations management': 2,
            'scheduling': 3, 'lean': 3,
        },
    },

    'A3_Customer_Service_Intelligence': {
        'name': 'Customer, Marketing & Service Intelligence',
        'description': 'Customer experience/service/engagement; CRM; personalization; '
                       'recommender systems & chatbots; sentiment, customer & marketing analytics',
        'keywords': [
            # Customer relationship
            'customer experience', 'customer service', 'customer satisfaction',
            'customer engagement', 'customer relationship management', 'CRM',
            'customer retention', 'customer loyalty', 'customer journey',
            # Personalization
            'personalization', 'personalisation', 'personalized marketing',
            # Conversational & recommender
            'recommendation system', 'recommendation systems', 'recommender', 'recommenders',
            'chatbot', 'chatbots', 'virtual assistant', 'virtual assistants',
            'conversational agent', 'conversational agents',
            # Analytics & behaviour
            'sentiment analysis', 'customer analytics', 'churn prediction',
            'service quality', 'service delivery', 'marketing analytics',
            'consumer behavior', 'consumer behaviour',
        ],
        'patterns': [
            r'(?:recommendation|recommender)\s+(?:system|engine)',
            r'\bchatbot\w*\b',
            r'virtual\s+assistant',
            r'conversational\s+(?:agent|AI)',
            r'churn\s+prediction',
            r'sentiment\s+analysis',
        ],
        'keyword_tiers': {
            'recommendation system': 1, 'recommendation systems': 1, 'recommender': 1,
            'recommenders': 1, 'chatbot': 1, 'chatbots': 1, 'virtual assistant': 1,
            'virtual assistants': 1, 'conversational agent': 1, 'conversational agents': 1,
            'churn prediction': 1, 'sentiment analysis': 1, 'personalized marketing': 1,
            'customer relationship management': 1, 'CRM': 1, 'customer analytics': 1,
            'marketing analytics': 1,
            'customer experience': 2, 'customer service': 2, 'customer engagement': 2,
            'customer retention': 2, 'customer loyalty': 2, 'customer journey': 2,
            'personalization': 2, 'personalisation': 2, 'service quality': 2,
            'service delivery': 2, 'consumer behavior': 2, 'consumer behaviour': 2,
            'customer satisfaction': 3,
        },
    },

    'A4_Product_Innovation': {
        'name': 'Product, Service & Innovation Management',
        'description': 'Product & service innovation; new product development; R&D; '
                       'AI-enabled & smart products; design automation; innovation ecosystems',
        'keywords': [
            # Innovation management
            'product innovation', 'service innovation', 'innovation management',
            'new product development', 'NPD', 'innovation capability',
            'innovation ecosystem', 'innovation ecosystems', 'open innovation',
            'digital innovation', 'technology innovation', 'innovation process',
            # R&D
            'R&D', 'research and development',
            # AI-enabled products
            'AI-enabled product', 'AI-enabled products', 'AI-enabled service', 'AI-enabled services',
            'smart product', 'smart products', 'intelligent product', 'intelligent products',
            # Design & development
            'generative design', 'design automation', 'product development', 'service development',
            'ideation',
        ],
        'patterns': [
            r'(?:product|service)\s+innovation',
            r'new\s+product\s+development',
            r'(?:research\s+and\s+development|R&D)',
            r'AI[\s-]?enabled\s+(?:product|service)',
            r'(?:smart|intelligent)\s+product',
            r'generative\s+design',
        ],
        'keyword_tiers': {
            'product innovation': 1, 'service innovation': 1, 'innovation management': 1,
            'new product development': 1, 'AI-enabled product': 1, 'AI-enabled products': 1,
            'AI-enabled service': 1, 'AI-enabled services': 1, 'smart product': 1,
            'smart products': 1, 'intelligent product': 1, 'intelligent products': 1,
            'generative design': 1, 'open innovation': 1, 'digital innovation': 1,
            'NPD': 2, 'innovation capability': 2, 'innovation ecosystem': 2,
            'innovation ecosystems': 2, 'technology innovation': 2, 'innovation process': 2,
            'design automation': 2, 'product development': 2, 'service development': 2,
            'research and development': 2,
            'R&D': 3, 'ideation': 3,
        },
    },

    'A5_Data_Decision_Intelligence': {
        'name': 'Data Analytics, Knowledge & Decision Intelligence',
        'description': 'Data/business/big-data analytics; business & decision intelligence; '
                       'predictive/prescriptive analytics; knowledge management; data & text mining',
        'keywords': [
            # Analytics
            'data analytics', 'business analytics', 'big data analytics', 'advanced analytics',
            'real-time analytics',
            # BI & decision intelligence
            'business intelligence', 'BI', 'decision intelligence',
            'decision support', 'decision support system', 'decision support systems', 'DSS',
            # Predictive / prescriptive
            'predictive analytics', 'prescriptive analytics', 'descriptive analytics',
            'forecasting', 'prediction',
            # Data-driven
            'data-driven decision', 'data-driven decisions', 'evidence-based decision',
            'evidence-based decisions',
            # Knowledge & mining
            'knowledge management', 'knowledge discovery', 'data mining', 'text mining',
            'anomaly detection', 'insight generation',
        ],
        'patterns': [
            r'(?:data|business|big\s+data)\s+analytics',
            r'(?:predictive|prescriptive|descriptive)\s+analytics',
            r'business\s+intelligence',
            r'decision\s+support\s+system',
            r'data[\s-]?driven\s+decision',
            r'(?:data|text)\s+mining',
        ],
        'keyword_tiers': {
            'predictive analytics': 1, 'prescriptive analytics': 1, 'big data analytics': 1,
            'business intelligence': 1, 'decision intelligence': 1, 'decision support system': 1,
            'decision support systems': 1, 'DSS': 1, 'knowledge discovery': 1, 'data mining': 1,
            'text mining': 1, 'anomaly detection': 1, 'real-time analytics': 1,
            'advanced analytics': 1,
            'data analytics': 2, 'business analytics': 2, 'descriptive analytics': 2,
            'decision support': 2, 'data-driven decision': 2, 'data-driven decisions': 2,
            'evidence-based decision': 2, 'evidence-based decisions': 2, 'knowledge management': 2,
            'insight generation': 2, 'BI': 2,
            'forecasting': 3, 'prediction': 3,
        },
    },

    'A6_Risk_Security_Governance': {
        'name': 'Risk, Security, Compliance & Governance',
        'description': 'Risk management & assessment; regulatory compliance; AI/data governance; '
                       'responsible/ethical/explainable AI; privacy, cybersecurity & fraud detection '
                       '(merges former Risk Management and AI Governance & Ethics)',
        'keywords': [
            # Risk
            'risk management', 'risk assessment', 'risk prediction', 'risk mitigation',
            'model risk', 'model monitoring', 'credit risk', 'operational risk', 'regulatory risk',
            # Compliance & governance
            'compliance', 'regulatory compliance', 'governance', 'AI governance',
            'data governance', 'algorithmic governance',
            # Responsible / ethical AI
            'responsible AI', 'trustworthy AI', 'ethical AI', 'AI ethics',
            'algorithmic accountability',
            # Fairness & explainability
            'bias detection', 'fairness', 'explainability', 'explainable AI', 'XAI', 'transparency',
            # Security & fraud
            'privacy', 'cybersecurity', 'fraud detection', 'anti-money laundering', 'AML',
        ],
        'patterns': [
            r'risk\s+(?:management|assessment|prediction|mitigation)',
            r'(?:regulatory\s+)?compliance',
            r'(?:AI|data|algorithmic)\s+governance',
            r'(?:responsible|trustworthy|ethical)\s+AI',
            r'explainab(?:le|ility)|interpretab(?:le|ility)|\bXAI\b',
            r'(?:fraud\s+detection|anti[\s-]?money\s+laundering|\bAML\b)',
        ],
        'keyword_tiers': {
            'risk management': 1, 'risk assessment': 1, 'AI governance': 1, 'data governance': 1,
            'responsible AI': 1, 'trustworthy AI': 1, 'ethical AI': 1, 'AI ethics': 1,
            'explainable AI': 1, 'XAI': 1, 'explainability': 1, 'bias detection': 1,
            'fraud detection': 1, 'anti-money laundering': 1, 'AML': 1,
            'algorithmic accountability': 1, 'model risk': 1, 'model monitoring': 1,
            'algorithmic governance': 1,
            'risk prediction': 2, 'risk mitigation': 2, 'regulatory compliance': 2,
            'credit risk': 2, 'operational risk': 2, 'regulatory risk': 2, 'cybersecurity': 2,
            'governance': 3, 'compliance': 3, 'fairness': 3, 'transparency': 3, 'privacy': 3,
        },
    },

    'A7_Human_Capital_Workforce': {
        'name': 'Human Capital, Workforce & Organizational Capability',
        'description': 'Human capital & talent management; HR/people/workforce analytics; '
                       'reskilling & upskilling; human-AI collaboration; future of work & change management',
        'keywords': [
            # Human capital & HR
            'human capital', 'workforce', 'talent management', 'workforce planning',
            'human resources', 'human resource', 'HR', 'HRM',
            # Analytics
            'people analytics', 'HR analytics', 'workforce analytics',
            # Talent lifecycle
            'recruitment', 'hiring', 'selection', 'employee performance',
            'employee productivity', 'employee engagement',
            # Skills & learning
            'skills', 'reskilling', 'upskilling', 'training',
            'organizational learning', 'organisational learning',
            # Capability & collaboration
            'digital capability', 'AI capability', 'human-AI collaboration',
            'human machine collaboration', 'future of work', 'job displacement',
            'workforce transformation', 'change management',
        ],
        'patterns': [
            r'(?:people|HR|workforce)\s+analytics',
            r'(?:re|up)skilling',
            r'human[\s-]?(?:AI|machine)\s+collaboration',
            r'(?:talent|workforce)\s+(?:management|planning|transformation)',
            r'change\s+management',
            r'future\s+of\s+work',
        ],
        'keyword_tiers': {
            'people analytics': 1, 'HR analytics': 1, 'workforce analytics': 1,
            'talent management': 1, 'workforce planning': 1, 'human-AI collaboration': 1,
            'human machine collaboration': 1, 'reskilling': 1, 'upskilling': 1,
            'workforce transformation': 1, 'AI capability': 1,
            'human capital': 2, 'workforce': 2, 'human resources': 2, 'human resource': 2,
            'employee performance': 2, 'employee productivity': 2, 'employee engagement': 2,
            'organizational learning': 2, 'organisational learning': 2, 'digital capability': 2,
            'future of work': 2, 'job displacement': 2, 'change management': 2, 'HRM': 2,
            'HR': 3, 'recruitment': 3, 'hiring': 3, 'selection': 3, 'skills': 3, 'training': 3,
        },
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# DIMENSION 2: AI TECHNOLOGIES & TECHNICAL CAPABILITIES
# ═══════════════════════════════════════════════════════════════════════════

AI_TECHNOLOGIES: dict[str, dict] = {
    'B1_Traditional_ML': {
        'name': 'Traditional Machine Learning',
        'description': 'Supervised/unsupervised learning; Ensemble methods; Statistical models',
        'keywords': [
            # Core ML — NO standalone "ML" (too ambiguous with "ML of water")
            'machine learning',  # full phrase only
            'supervised learning', 'unsupervised learning',
            'classification', 'regression', 'clustering',

            # Algorithms — HIGH CONFIDENCE (Tier 1)
            'random forest', 'decision tree', 'logistic regression',
            'support vector machine', 'SVM', 'k-means', 'XGBoost',
            'gradient boosting', 'ensemble', 'bagging', 'boosting',

            # Process — MEDIUM CONFIDENCE (Tier 2)
            'feature engineering', 'model training', 'cross-validation',
            'hyperparameter tuning', 'overfitting', 'underfitting',
        ],
        'patterns': [
            r'machine\s+learning(?!\s+(?:operation|platform|infrastructure))',
            r'\bML\b(?!\s*(?:Ops|infrastructure|platform|of|metric|tons?|liters?|gallons?|\d))',
            r'(?:supervised|unsupervised)\s+learning',
            r'(?:random\s+forest|decision\s+tree|XGBoost)',
            r'(?:classification|regression|clustering)\s+(?:model|algorithm)',
            r'ML\s+(?:model|algorithm|technique|approach|system)',  # ML only with technical context
            r'(?:train|build|develop).*ML.*(?:model|algorithm)',
        ],
        'keyword_tiers': {
            'machine learning': 1, 'supervised learning': 1, 'unsupervised learning': 1,
            'random forest': 1, 'decision tree': 1, 'XGBoost': 1,
            'gradient boosting': 1, 'support vector machine': 1, 'SVM': 1,
            'classification': 2, 'regression': 2, 'clustering': 2,
            'ensemble': 2, 'bagging': 2, 'boosting': 2,
            'feature engineering': 3, 'model training': 3, 'cross-validation': 3,
            'hyperparameter tuning': 3, 'overfitting': 3, 'underfitting': 3,
        },
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
            r'\bDL\b(?!\s+(?:of|metric))',  # exclude "DL of water"
            r'(?:deep\s+)?neural\s+network',
            r'\b(?:CNN|RNN|LSTM|GRU)\b',
            r'(?:convolutional|recurrent)\s+neural',
            r'reinforcement\s+learning',
        ],
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
        ],
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
            r'\b(?:Claude|Gemini|Bard|PaLM)\b(?!\s+(?:project|program))',  # exclude "Gemini project"
            r'foundation\s+model(?:s)?',
            r'\bGenAI\b',
            r'(?:retrieval[\s-]?augmented|RAG)',
            r'(?:vector|embedding)\s+(?:database|store)',
            r'(?:prompt|instruction)\s+(?:engineering|tuning)',
            r'(?:text|image|code)\s+generation',
        ],
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
        ],
    },

    'B6_Robotics_Autonomous': {
        'name': 'Robotics & Autonomous Systems',
        'description': 'Autonomous vehicles; AI-powered robotics; Agentic AI workflows '
                       '(not generic copilot agents); Multi-agent systems',
        'keywords': [
            # Autonomous systems — HIGH CONFIDENCE (Tier 1)
            'autonomous', 'self-driving', 'autonomous vehicle',
            'autonomous navigation', 'path planning',

            # Robotics (AI-powered) — HIGH CONFIDENCE (Tier 1)
            'intelligent robot', 'cognitive robot', 'autonomous robot',
            'collaborative robot', 'cobot', 'robot learning',

            # Agentic AI — SPECIFIC to autonomous workflows
            'agentic', 'AI agent', 'autonomous agent',
            'multi-agent', 'agent orchestration', 'agent framework',
            'tool use', 'function calling', 'computer use',

            # Specific systems — HIGH CONFIDENCE (Tier 1)
            'drone', 'UAV', 'unmanned aerial', 'AGV', 'automated guided vehicle',
        ],
        'patterns': [
            r'(?:autonomous|self[\s-]?driving)\s+(?:vehicle|system|car)',
            r'(?:agentic)\s+(?:system|AI|workflow)',
            r'(?:autonomous\s+)?AI\s+agent(?:s)?',
            r'multi[\s-]?agent\s+(?:system|orchestration)',
            r'agent\s+(?:orchestration|framework|coordination)',
            r'(?:tool|function)\s+(?:use|calling)',
            r'(?:drone|UAV|AGV).*(?:AI|autonomous|intelligent)',
            r'(?:cognitive|intelligent|autonomous)\s+robot',
            r'(?:collaborative\s+)?robot.*(?:AI|learning|intelligent)',
        ],
        'keyword_tiers': {
            'autonomous': 1, 'self-driving': 1, 'autonomous vehicle': 1,
            'intelligent robot': 1, 'cognitive robot': 1, 'autonomous robot': 1,
            'agentic': 1, 'multi-agent': 1, 'drone': 1, 'UAV': 1, 'AGV': 1,
            'AI agent': 2, 'autonomous agent': 2, 'cobot': 2,
            'agent orchestration': 2, 'agent framework': 2,
            'tool use': 3, 'function calling': 3, 'computer use': 3,
            'path planning': 3, 'robot learning': 3,
        },
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
        ],
    },

    'B8_General_AI': {
        'name': 'AI (General/Unspecified)',
        'description': 'Generic AI references without technical specificity',
        'keywords': [
            'artificial intelligence', 'AI', 'AI system', 'AI solution',
            'AI technology', 'AI capability', 'intelligent system',
            'cognitive', 'smart', 'intelligent',
        ],
        'patterns': [
            r'\bartificial\s+intelligence\b',
            r'\bAI\b(?!\s*(?:powered|enabled|first))',  # AI without specific context
            r'AI\s+(?:system|solution|technology|capability)',
        ],
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# EU_SEMANTICS TAXONOMY — European Commission JRC "AI Watch"
# ═══════════════════════════════════════════════════════════════════════════
# Domains / subdomains with characteristic keywords. Classified IN PARALLEL with
# the classic taxonomy: every detected reference also receives an EU domain +
# subdomain. EU keywords default to Tier 1 (specific technical vocabulary);
# 'keyword_tiers' lists only the generic terms downgraded to Tier 2/3.

EU_SEMANTICS: dict[str, dict] = {
    'E1_Reasoning': {
        'domain': 'Reasoning',
        'domain_code': 'E1',
        'subdomain': 'Knowledge representation; Automated reasoning; Common sense reasoning',
        'keywords': [
            'case-based reasoning', 'causal inference', 'causal models',
            'common-sense reasoning', 'expert system', 'fuzzy logic', 'graphical models',
            'inductive programming', 'information theory', 'knowledge representation',
            'knowledge representation & reasoning', 'latent variable models', 'semantic web',
            'uncertainty in artificial intelligence',
        ],
        'patterns': [],
        'keyword_tiers': {'information theory': 2},
    },

    'E2_Planning': {
        'domain': 'Planning',
        'domain_code': 'E2',
        'subdomain': 'Planning and Scheduling; Searching; Optimisation',
        'keywords': [
            'bayesian optimisation', 'constraint satisfaction', 'evolutionary algorithm',
            'genetic algorithm', 'gradient descent', 'hierarchical task network',
            'metaheuristic optimisation', 'planning graph', 'stochastic optimisation',
        ],
        'patterns': [],
        'keyword_tiers': {},
    },

    'E3_Learning': {
        'domain': 'Learning',
        'domain_code': 'E3',
        'subdomain': 'Machine learning',
        'keywords': [
            'active learning', 'adaptive learning', 'adversarial machine learning',
            'adversarial network', 'anomaly detection', 'artificial neural network',
            'automated machine learning', 'automatic classification', 'automatic recognition',
            'bagging', 'bayesian modelling', 'boosting', 'classification', 'clustering',
            'collaborative filtering', 'content-based filtering', 'convolutional neural network',
            'data mining', 'deep learning', 'deep neural network', 'ensemble method',
            'feature extraction', 'generative adversarial network', 'generative model',
            'multi-task learning', 'neural network', 'pattern recognition', 'probabilistic learning',
            'probabilistic model', 'recommender system', 'recurrent neural network',
            'recursive neural network', 'reinforcement learning', 'semi-supervised learning',
            'statistical learning', 'statistical relational learning', 'supervised learning',
            'support vector machine', 'transfer learning', 'unstructured data', 'unsupervised learning',
        ],
        'patterns': [],
        'keyword_tiers': {'classification': 2, 'clustering': 2, 'boosting': 2, 'bagging': 2},
    },

    'E4_Communication': {
        'domain': 'Communication',
        'domain_code': 'E4',
        'subdomain': 'Natural language processing',
        'keywords': [
            'chatbot', 'computational linguistics', 'conversation model', 'coreference resolution',
            'information extraction', 'information retrieval', 'natural language understanding',
            'natural language generation', 'machine translation', 'question answering',
            'sentiment analysis', 'text classification', 'text mining',
        ],
        'patterns': [],
        'keyword_tiers': {},
    },

    'E5a_Computer_Vision': {
        'domain': 'Perception',
        'domain_code': 'E5',
        'subdomain': 'Computer vision',
        'keywords': [
            'action recognition', 'face recognition', 'gesture recognition', 'image processing',
            'image retrieval', 'object recognition', 'recognition technology', 'sensor network',
            'visual search',
        ],
        'patterns': [],
        'keyword_tiers': {'image processing': 2, 'sensor network': 2},
    },

    'E5b_Audio_Processing': {
        'domain': 'Perception',
        'domain_code': 'E5',
        'subdomain': 'Audio processing',
        'keywords': [
            'computational auditory scene analysis', 'music information retrieval',
            'sound description', 'sound event recognition', 'sound source separation',
            'sound synthesis', 'speaker identification', 'speech processing', 'speech recognition',
            'speech synthesis',
        ],
        'patterns': [],
        'keyword_tiers': {},
    },

    'E6a_Multi_Agent': {
        'domain': 'Integration and Interaction',
        'domain_code': 'E6',
        'subdomain': 'Multi-agent systems',
        'keywords': [
            'agent-based modelling', 'agreement technologies', 'computational economics',
            'game theory', 'intelligent agent', 'negotiation algorithm', 'network intelligence',
            'q-learning', 'swarm intelligence',
        ],
        'patterns': [],
        'keyword_tiers': {'game theory': 2},
    },

    'E6b_Robotics_Automation': {
        'domain': 'Integration and Interaction',
        'domain_code': 'E6',
        'subdomain': 'Robotics and Automation',
        'keywords': [
            'cognitive system', 'control theory', 'human-ai interaction', 'industrial robot',
            'robot system', 'service robot', 'social robot',
        ],
        'patterns': [],
        'keyword_tiers': {'cognitive system': 2, 'control theory': 2},
    },

    'E6c_Connected_Vehicles': {
        'domain': 'Integration and Interaction',
        'domain_code': 'E6',
        'subdomain': 'Connected and Automated vehicles',
        'keywords': [
            'autonomous driving', 'autonomous system', 'autonomous vehicle', 'self-driving car',
            'unmanned vehicle',
        ],
        'patterns': [],
        'keyword_tiers': {},
    },

    'E7_Services': {
        'domain': 'Services',
        'domain_code': 'E7',
        'subdomain': 'AI Services',
        'keywords': [
            'ai application', 'ai benchmark', 'ai competition', 'ai software toolkit',
            'analytics platform', 'big data', 'business intelligence', 'central processing unit',
            'computational creativity', 'computational neuroscience', 'data analytics',
            'decision analytics', 'decision support', 'distributed computing',
            'graphics processing unit', 'intelligence software', 'intelligent control',
            'intelligent control system', 'intelligent hardware development',
            'intelligent software development', 'intelligent user interface', 'internet of things',
            'machine learning framework', 'machine learning library', 'machine learning platform',
            'personal assistant', 'platform as a service', 'tensor processing unit',
            'virtual environment', 'virtual reality',
        ],
        'patterns': [],
        'keyword_tiers': {
            'data analytics': 2, 'business intelligence': 2, 'decision support': 2,
            'distributed computing': 2, 'internet of things': 2, 'graphics processing unit': 2,
            'virtual reality': 2, 'big data': 3, 'central processing unit': 3,
            'virtual environment': 3,
        },
    },

    'E8a_AI_Ethics': {
        'domain': 'AI Ethics and Philosophy',
        'domain_code': 'E8',
        'subdomain': 'AI Ethics',
        'keywords': [
            'accountability', 'explainability', 'fairness', 'privacy', 'safety', 'security',
            'transparency',
        ],
        'patterns': [],
        'keyword_tiers': {
            'fairness': 2, 'accountability': 2, 'transparency': 3, 'privacy': 3,
            'safety': 3, 'security': 3,
        },
    },

    'E8b_Philosophy_AI': {
        'domain': 'AI Ethics and Philosophy',
        'domain_code': 'E8',
        'subdomain': 'Philosophy of AI',
        'keywords': [
            'artificial general intelligence', 'strong artificial intelligence',
            'weak artificial intelligence', 'narrow artificial intelligence',
        ],
        'patterns': [],
        'keyword_tiers': {},
    },
}


# ═══════════════════════════════════════════════════════════════════════════
# FALLBACK CATEGORY NAMES → CLASSIC CODES
# ═══════════════════════════════════════════════════════════════════════════
# The classifier's fallback rules emit human-readable category names; this map
# translates those names into the canonical classic category codes.

FALLBACK_NAME_TO_CODE: dict[str, str] = {
    # Applications
    'Product Development': 'A4_Product_Innovation',
    'Research & Innovation': 'A4_Product_Innovation',
    'Operational Implementation': 'A2_Operational_Optimization',
    'Customer Experience': 'A3_Customer_Service_Intelligence',
    'Risk & Compliance': 'A6_Risk_Security_Governance',
    'Data & Analytics': 'A5_Data_Decision_Intelligence',
    'Strategic Investment': 'A1_Strategic_Transformation',
    'Explainability & AI Safety': 'A6_Risk_Security_Governance',
    'Talent & Workforce': 'A7_Human_Capital_Workforce',

    # Technologies
    'Generative AI & LLMs': 'B4_GenAI_LLMs',
    'AI Infrastructure & MLOps': 'B7_Infrastructure_Platforms',
    'AI Coding & Development': 'B7_Infrastructure_Platforms',
    'Agentic AI Systems': 'B6_Robotics_Autonomous',
}


# ═══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def get_all_application_keywords() -> list[str]:
    """Return all keywords from every application category (deduplicated)."""
    keywords: list[str] = []
    for category in AI_APPLICATIONS.values():
        keywords.extend(category['keywords'])
    return list(set(keywords))


def get_all_technology_keywords() -> list[str]:
    """Return all keywords from every technology category (deduplicated)."""
    keywords: list[str] = []
    for category in AI_TECHNOLOGIES.values():
        keywords.extend(category['keywords'])
    return list(set(keywords))


def get_category_info(category_code: str) -> dict | None:
    """Return metadata for a category code (application or technology)."""
    if category_code in AI_APPLICATIONS:
        return AI_APPLICATIONS[category_code]
    if category_code in AI_TECHNOLOGIES:
        return AI_TECHNOLOGIES[category_code]
    return None


def get_keyword_tier(category_code: str, keyword: str) -> int:
    """Return the confidence tier of a keyword.

    Returns 1 (high), 2 (medium), or 3 (low). Defaults to 2 if the keyword
    has no explicit tier or the category is unknown.
    """
    category = get_category_info(category_code)
    if not category:
        return 2
    tiers: dict[str, int] = category.get('keyword_tiers', {})
    return tiers.get(keyword, 2)


def get_high_confidence_keywords(category_code: str) -> list[str]:
    """Return only Tier 1 (high-confidence) keywords for a category."""
    category = get_category_info(category_code)
    if not category:
        return []
    tiers: dict[str, int] = category.get('keyword_tiers', {})
    return [kw for kw, tier in tiers.items() if tier == 1]


# ─── EU_SEMANTICS helpers ───────────────────────────────────────────────────

EU_DOMAINS: list[tuple[str, str]] = [
    ('E1', 'Reasoning'),
    ('E2', 'Planning'),
    ('E3', 'Learning'),
    ('E4', 'Communication'),
    ('E5', 'Perception'),
    ('E6', 'Integration and Interaction'),
    ('E7', 'Services'),
    ('E8', 'AI Ethics and Philosophy'),
]

EU_LEAF_UNITS: list[str] = list(EU_SEMANTICS.keys())


def get_eu_info(leaf_code: str) -> dict | None:
    """Return metadata for an EU_SEMANTICS leaf code (e.g. 'E3_Learning')."""
    return EU_SEMANTICS.get(leaf_code)


def get_eu_keyword_tier(leaf_code: str, keyword: str) -> int:
    """Return the confidence tier (1/2/3) of an EU keyword.

    EU keywords default to Tier 1 (specific technical vocabulary); only generic
    terms are explicitly downgraded in each leaf's 'keyword_tiers'.
    """
    leaf = EU_SEMANTICS.get(leaf_code)
    if not leaf:
        return 1
    return leaf.get('keyword_tiers', {}).get(keyword, 1)


def get_all_eu_keywords() -> list[str]:
    """Return all EU_SEMANTICS keywords (deduplicated)."""
    keywords: list[str] = []
    for leaf in EU_SEMANTICS.values():
        keywords.extend(leaf['keywords'])
    return list(set(keywords))


def print_taxonomy_summary() -> None:
    """Print a human-readable summary of the semantic taxonomies."""
    print("=" * 80)
    print("SEMANTIC TAXONOMIES - SUMMARY WITH KEYWORD TIERS")
    print("=" * 80)

    print("\n[CLASSIC] DIMENSION 1: AI APPLICATIONS (7 categories)")
    print("-" * 80)
    for code, info in AI_APPLICATIONS.items():
        total_kw = len(info['keywords'])
        tiers: dict[str, int] = info.get('keyword_tiers', {})
        tier1 = sum(1 for t in tiers.values() if t == 1)
        tier2 = sum(1 for t in tiers.values() if t == 2)
        tier3 = sum(1 for t in tiers.values() if t == 3)

        print(f"{code}: {info['name']}")
        print(f"   Total: {total_kw} keywords, {len(info['patterns'])} patterns")
        if tiers:
            print(f"   Tiers: {tier1} high, {tier2} medium, {tier3} low")

    print("\n[CLASSIC] DIMENSION 2: AI TECHNOLOGIES (8 categories)")
    print("-" * 80)
    for code, info in AI_TECHNOLOGIES.items():
        total_kw = len(info['keywords'])
        tiers = info.get('keyword_tiers', {})
        tier1 = sum(1 for t in tiers.values() if t == 1)
        tier2 = sum(1 for t in tiers.values() if t == 2)
        tier3 = sum(1 for t in tiers.values() if t == 3)

        print(f"{code}: {info['name']}")
        print(f"   Total: {total_kw} keywords, {len(info['patterns'])} patterns")
        if tiers:
            print(f"   Tiers: {tier1} high, {tier2} medium, {tier3} low")

    print("\n[EU_SEMANTICS] DOMAINS / SUBDOMAINS (8 domains, 12 leaves)")
    print("-" * 80)
    for code, leaf in EU_SEMANTICS.items():
        print(f"{code}  [{leaf['domain']}] {leaf['subdomain']}")
        print(f"   Total: {len(leaf['keywords'])} keywords")

    total_app_keywords = len(get_all_application_keywords())
    total_tech_keywords = len(get_all_technology_keywords())
    total_eu_keywords = len(get_all_eu_keywords())

    print("\n" + "=" * 80)
    print(f"Total unique application keywords: {total_app_keywords}")
    print(f"Total unique technology keywords: {total_tech_keywords}")
    print(f"Total unique EU_Semantics keywords: {total_eu_keywords}")
    print("Classic categories: 15 (7 applications + 8 technologies)")
    print("EU_Semantics: 8 domains / 12 subdomain leaves")
    print("\nNOTES:")
    print("  - Dimension 1: 7 business categories (A1-A7)")
    print("  - Dimension 2: 8 technology categories (B1-B8)")
    print("  - EU_SEMANTICS taxonomy (European Commission JRC AI Watch)")
    print("  - Each reference is labelled under BOTH taxonomies in one run")
    print("=" * 80)


if __name__ == "__main__":
    print_taxonomy_summary()
