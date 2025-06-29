# Ilanya Desire Engine - Architecture

## Engine Overview

```mermaid
graph TB
    subgraph "Desire Engine Core"
        subgraph "Main Engine"
            DE[DesireEngine]
            CONFIG[Configuration]
            STATE[State Management]
        end
        
        subgraph "Core Modules"
            IM[Interaction Module]
            TM[Threshold Module]
            EM[Embedding Module]
            AM[Attention Module]
            IM2[Information Module]
            TEM[Temporal Module]
        end
        
        subgraph "Data Structures"
            DESIRES[Desire Objects]
            INTERACTIONS[Interaction Network]
            EMERGENT[Emergent Desires]
            METRICS[System Metrics]
        end
        
        subgraph "External Interfaces"
            TRAITS[Trait Activations]
            GOALS[Goal Generation]
            LOGS[Logging System]
        end
    end
    
    %% Core connections
    DE --> CONFIG
    DE --> STATE
    DE --> DESIRES
    DE --> INTERACTIONS
    DE --> EMERGENT
    DE --> METRICS
    
    %% Module connections
    DE --> IM
    DE --> TM
    DE --> EM
    DE --> AM
    DE --> IM2
    DE --> TEM
    
    %% Data flow
    TRAITS --> DE
    DE --> GOALS
    DE --> LOGS
    
    %% Module interactions
    IM --> INTERACTIONS
    TM --> EMERGENT
    EM --> DESIRES
    AM --> DESIRES
    IM2 --> METRICS
    TEM --> STATE
```

## Module Details

### 1. Core Engine Components

#### **DesireEngine (Main Controller)**
```python
class DesireEngine:
    - desires: Dict[str, Desire]
    - interaction_module: InteractionModule
    - threshold_module: ThresholdModule
    - embedding_module: EmbeddingModule
    - attention_module: AttentionModule
    - information_module: InformationModule
    - temporal_module: TemporalModule
```

#### **Configuration Management**
```python
class DesireEngineConfig:
    - interaction_threshold: float
    - synergy_threshold: float
    - emergent_threshold: float
    - conflict_threshold: float
    - decay_rate: float
    - reinforcement_rate: float
```

### 2. Processing Modules

#### **Interaction Module**
```mermaid
graph LR
    A[Desire Pairs] --> B[Calculate Synergy]
    B --> C[Create Emergent Desires]
    C --> D[Update Network]
    D --> E[Log Interactions]
```

**Key Functions:**
- `process_interactions()`: Process all desire pairs
- `_calculate_interaction_strength()`: Compute synergy/conflict
- `_create_emergent_desire()`: Generate new emergent desires

#### **Threshold Module**
```mermaid
graph LR
    A[Desire Strength] --> B[Check Thresholds]
    B --> C{Above Threshold?}
    C -->|Yes| D[Activate Desire]
    C -->|No| E[Deactivate Desire]
```

**Key Functions:**
- `apply_thresholds()`: Filter desires by strength
- `prune_weak_desires()`: Remove low-strength desires
- `reinforce_strong_desires()`: Boost high-strength desires

#### **Embedding Module**
```mermaid
graph LR
    A[Desire Data] --> B[Create Embeddings]
    B --> C[Compute Similarities]
    C --> D[Update Representations]
```

**Key Functions:**
- `create_embeddings()`: Generate vector representations
- `compute_similarities()`: Calculate desire similarities
- `update_embeddings()`: Refresh embeddings

### 3. Data Structures

#### **Desire Object**
```python
@dataclass
class Desire:
    id: str
    name: str
    source_traits: List[str]
    strength: float
    base_strength: float
    reinforcement_count: int
    last_reinforcement: datetime
    emergent: bool = False
    parent_desires: List[str] = None
```

#### **Interaction Network**
```python
class InteractionNetwork:
    - interactions: Dict[Tuple[str, str], float]
    - emergent_desires: List[Desire]
    - synergy_matrix: np.ndarray
    - conflict_matrix: np.ndarray
```

### 4. Processing Pipeline

```mermaid
sequenceDiagram
    participant User
    participant DesireEngine
    participant InteractionModule
    participant ThresholdModule
    participant EmbeddingModule
    
    User->>DesireEngine: Process Traits
    DesireEngine->>DesireEngine: Reinforce Desires
    DesireEngine->>InteractionModule: Process Interactions
    InteractionModule->>DesireEngine: Create Emergent Desires
    DesireEngine->>ThresholdModule: Apply Thresholds
    ThresholdModule->>DesireEngine: Prune Weak Desires
    DesireEngine->>EmbeddingModule: Update Embeddings
    EmbeddingModule->>DesireEngine: Return Updated State
    DesireEngine->>User: Return Results
```

## Key Features

### 🎯 **Emergent Desire Creation**
- Automatic generation of new desires from interactions
- Synergy-based emergent behavior
- Hierarchical desire relationships

### 🔄 **Dynamic Processing**
- Real-time desire strength updates
- Adaptive threshold management
- Continuous interaction processing

### 📊 **Comprehensive Metrics**
- Entropy calculations
- Complexity measurements
- Stability tracking
- Network analysis

### 🧠 **Neural Integration**
- Embedding-based desire representation
- Attention mechanisms for focus
- Temporal processing for evolution

## File Structure

```
IlanyaDesireEngine/
├── desire_engine/
│   ├── __init__.py
│   ├── core.py              # Main engine
│   ├── config.py            # Configuration
│   ├── models.py            # Data models
│   └── modules/
│       ├── __init__.py
│       ├── attention.py     # Attention mechanisms
│       ├── embedding.py     # Embedding generation
│       ├── information.py   # Information processing
│       ├── interaction.py   # Interaction handling
│       ├── temporal.py      # Temporal processing
│       └── threshold.py     # Threshold management
├── requirements.txt
└── README.md
```

## Configuration Options

| Parameter | Default | Description |
|-----------|---------|-------------|
| `interaction_threshold` | 0.05 | Minimum interaction strength |
| `synergy_threshold` | 0.2 | Synergy detection threshold |
| `emergent_threshold` | 0.3 | Emergent desire creation threshold |
| `conflict_threshold` | 0.1 | Conflict detection threshold |
| `decay_rate` | 0.01 | Desire strength decay rate |
| `reinforcement_rate` | 0.1 | Reinforcement learning rate | 