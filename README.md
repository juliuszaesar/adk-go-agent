# OpenRouter Wrapper for Google ADK-Go

A custom OpenRouter model wrapper that implements the Google Agent Development Kit (ADK) `model.LLM` interface, enabling you to use any OpenRouter-compatible model with Google's ADK framework.

## Features

- ✅ **Universal Model Support**: Use any model available on OpenRouter (OpenAI, Anthropic, X.AI, Meta, etc.)
- ✅ **Full Tool Calling**: Complete support for function/tool calling with proper format conversion
- ✅ **Streaming & Non-Streaming**: Supports both response modes
- ✅ **ADK Compatible**: Implements the official `google.golang.org/adk/model.LLM` interface
- ✅ **Configuration Options**: Temperature, top_p, max_tokens, stop sequences
- ✅ **Usage Metadata**: Returns token usage information

## Prerequisites

- Go 1.23 or higher (requires `iter.Seq2` support)
- OpenRouter API key ([get one here](https://openrouter.ai/keys))

## Installation

1. Clone this repository:
```bash
git clone https://github.com/YOUR_USERNAME/adk-go.git
cd adk-go
```

2. Install dependencies:
```bash
go mod download
```

3. Create a `.env` file with your OpenRouter API key:
```bash
echo "OPENROUTER_API_KEY=your_api_key_here" > .env
```

## Usage

### Basic Example

```go
package main

import (
    "context"
    "log"
    "os"
    
    "google.golang.org/adk/agent"
    "google.golang.org/adk/agent/llmagent"
    "google.golang.org/adk/cmd/launcher"
    "google.golang.org/adk/cmd/launcher/full"
)

func main() {
    ctx := context.Background()
    
    // Create OpenRouter model
    model, err := NewOpenRouterModel("x-ai/grok-code-fast-1", &OpenRouterConfig{
        APIKey: os.Getenv("OPENROUTER_API_KEY"),
    })
    if err != nil {
        log.Fatalf("Failed to create model: %v", err)
    }
    
    // Create agent
    agent, err := llmagent.New(llmagent.Config{
        Name:        "my_agent",
        Model:       model,
        Description: "A helpful AI assistant",
        Instruction: "You are a helpful assistant.",
    })
    if err != nil {
        log.Fatalf("Failed to create agent: %v", err)
    }
    
    // Launch
    config := &launcher.Config{
        AgentLoader: agent.NewSingleLoader(agent),
    }
    
    l := full.NewLauncher()
    if err = l.Execute(ctx, config, os.Args[1:]); err != nil {
        log.Fatalf("Run failed: %v", err)
    }
}
```

### Available Models

You can use any model from OpenRouter. Popular examples:

- `openai/gpt-4-turbo`
- `anthropic/claude-3-opus`
- `x-ai/grok-code-fast-1`
- `meta-llama/llama-3.1-70b-instruct`
- `google/gemini-pro-1.5`

See the full list at [openrouter.ai/models](https://openrouter.ai/models)

### Custom Configuration

```go
model, err := NewOpenRouterModel("openai/gpt-4-turbo", &OpenRouterConfig{
    APIKey:  os.Getenv("OPENROUTER_API_KEY"),
    BaseURL: "https://openrouter.ai/api/v1", // Optional, this is the default
})
```

## Running the Agent

```bash
# Set your API key
export OPENROUTER_API_KEY=your_key_here

# Build
go build -o agent

# Run in console mode
./agent console

# Run in web mode
./agent web
```

## Project Structure

- `agent.go` - Main application entry point
- `openrouter_model.go` - OpenRouter wrapper implementing `model.LLM` interface
- `go.mod` - Go module dependencies

## Important Notes

### Tool Compatibility

⚠️ **Gemini-native tools like `geminitool.GoogleSearch` will NOT work** with OpenRouter models. These are built-in capabilities of Google's Gemini models.

For OpenRouter models, you need to:
- Use `functiontool` to create custom function tools
- Implement your own search API integration (SerpAPI, Tavily, Brave Search, etc.)
- Or use models that have built-in web search capabilities

### Environment Variables

Make sure to set your OpenRouter API key:
```bash
export OPENROUTER_API_KEY=your_key_here
```

Or use a `.env` file (already in `.gitignore`).

## How It Works

The wrapper converts between Google ADK's format and OpenAI-compatible format:

1. **Request Conversion**: Converts `genai.Content` messages to OpenAI `ChatCompletionMessage`
2. **Tool Conversion**: Converts `genai.FunctionDeclaration` to OpenAI `Tool` format
3. **Response Conversion**: Converts OpenAI responses back to `genai.Content`
4. **Streaming**: Handles Server-Sent Events (SSE) for streaming responses

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

- Built on [Google ADK-Go](https://github.com/google/adk-go)
- Uses [go-openai](https://github.com/sashabaranov/go-openai) for API communication
- Powered by [OpenRouter](https://openrouter.ai)

