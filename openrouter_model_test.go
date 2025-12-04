package main

import (
	"testing"

	"github.com/sashabaranov/go-openai"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// ============================================================================
// NewOpenRouterModel Tests
// ============================================================================

func TestNewOpenRouterModel_Success(t *testing.T) {
	model, err := NewOpenRouterModel("openai/gpt-4", &OpenRouterConfig{
		APIKey: "test-api-key",
	})

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if model == nil {
		t.Fatal("expected model to be non-nil")
	}
	if model.modelName != "openai/gpt-4" {
		t.Errorf("expected model name 'openai/gpt-4', got '%s'", model.modelName)
	}
}

func TestNewOpenRouterModel_WithCustomBaseURL(t *testing.T) {
	model, err := NewOpenRouterModel("anthropic/claude-3", &OpenRouterConfig{
		APIKey:  "test-api-key",
		BaseURL: "https://custom.api.endpoint/v1",
	})

	if err != nil {
		t.Fatalf("expected no error, got %v", err)
	}
	if model == nil {
		t.Fatal("expected model to be non-nil")
	}
}

func TestNewOpenRouterModel_NilConfig(t *testing.T) {
	model, err := NewOpenRouterModel("openai/gpt-4", nil)

	if err == nil {
		t.Fatal("expected error for nil config")
	}
	if model != nil {
		t.Error("expected model to be nil on error")
	}
	if err.Error() != "OpenRouter API key is required" {
		t.Errorf("unexpected error message: %v", err)
	}
}

func TestNewOpenRouterModel_EmptyAPIKey(t *testing.T) {
	model, err := NewOpenRouterModel("openai/gpt-4", &OpenRouterConfig{
		APIKey: "",
	})

	if err == nil {
		t.Fatal("expected error for empty API key")
	}
	if model != nil {
		t.Error("expected model to be nil on error")
	}
}

// ============================================================================
// Name() Tests
// ============================================================================

func TestName(t *testing.T) {
	tests := []struct {
		name      string
		modelName string
	}{
		{"OpenAI model", "openai/gpt-4"},
		{"Anthropic model", "anthropic/claude-3-opus"},
		{"X.AI model", "x-ai/grok-3-fast-beta"},
		{"Meta model", "meta-llama/llama-3.1-70b"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			model := &OpenRouterModel{modelName: tt.modelName}
			if got := model.Name(); got != tt.modelName {
				t.Errorf("Name() = %q, want %q", got, tt.modelName)
			}
		})
	}
}

// ============================================================================
// convertRole Tests
// ============================================================================

func TestConvertRole(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"user", openai.ChatMessageRoleUser},
		{"model", openai.ChatMessageRoleAssistant},
		{"assistant", openai.ChatMessageRoleAssistant},
		{"system", openai.ChatMessageRoleSystem},
		{"tool", openai.ChatMessageRoleTool},
		{"unknown", openai.ChatMessageRoleUser},
		{"", openai.ChatMessageRoleUser},
		{"USER", openai.ChatMessageRoleUser}, // case sensitive - defaults to user
	}

	for _, tt := range tests {
		t.Run(tt.input, func(t *testing.T) {
			result := convertRole(tt.input)
			if result != tt.expected {
				t.Errorf("convertRole(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

// ============================================================================
// convertFinishReason Tests
// ============================================================================

func TestConvertFinishReason(t *testing.T) {
	tests := []struct {
		name     string
		input    openai.FinishReason
		expected genai.FinishReason
	}{
		{"stop", openai.FinishReasonStop, genai.FinishReasonStop},
		{"length", openai.FinishReasonLength, genai.FinishReasonMaxTokens},
		{"tool_calls", openai.FinishReasonToolCalls, genai.FinishReasonStop},
		{"function_call", openai.FinishReasonFunctionCall, genai.FinishReasonStop},
		{"unknown", openai.FinishReason("unknown"), genai.FinishReasonUnspecified},
		{"empty", openai.FinishReason(""), genai.FinishReasonUnspecified},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := convertFinishReason(tt.input)
			if result != tt.expected {
				t.Errorf("convertFinishReason(%q) = %v, want %v", tt.input, result, tt.expected)
			}
		})
	}
}

// ============================================================================
// joinStrings Tests
// ============================================================================

func TestJoinStrings(t *testing.T) {
	tests := []struct {
		name     string
		input    []string
		expected string
	}{
		{"empty slice", []string{}, ""},
		{"single string", []string{"hello"}, "hello"},
		{"multiple strings", []string{"hello", " ", "world"}, "hello world"},
		{"no separator", []string{"a", "b", "c"}, "abc"},
		{"with empty strings", []string{"a", "", "b"}, "ab"},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := joinStrings(tt.input)
			if result != tt.expected {
				t.Errorf("joinStrings(%v) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

// ============================================================================
// extractText Tests
// ============================================================================

func TestExtractText(t *testing.T) {
	tests := []struct {
		name     string
		content  *genai.Content
		expected string
	}{
		{
			name: "single text part",
			content: &genai.Content{
				Parts: []*genai.Part{
					genai.NewPartFromText("Hello, world!"),
				},
			},
			expected: "Hello, world!",
		},
		{
			name: "multiple text parts",
			content: &genai.Content{
				Parts: []*genai.Part{
					genai.NewPartFromText("Hello, "),
					genai.NewPartFromText("world!"),
				},
			},
			expected: "Hello, world!",
		},
		{
			name: "empty parts",
			content: &genai.Content{
				Parts: []*genai.Part{},
			},
			expected: "",
		},
		{
			name:     "nil parts",
			content:  &genai.Content{},
			expected: "",
		},
		{
			name: "mixed parts with function call",
			content: &genai.Content{
				Parts: []*genai.Part{
					genai.NewPartFromText("Processing..."),
					genai.NewPartFromFunctionCall("get_weather", map[string]any{"city": "London"}),
				},
			},
			expected: "Processing...",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := extractText(tt.content)
			if result != tt.expected {
				t.Errorf("extractText() = %q, want %q", result, tt.expected)
			}
		})
	}
}

// ============================================================================
// convertSchema Tests
// ============================================================================

func TestConvertSchema_Simple(t *testing.T) {
	schema := &genai.Schema{
		Type:        "string",
		Description: "A simple string",
	}

	result := convertSchema(schema)

	if result["type"] != "string" {
		t.Errorf("expected type 'string', got %v", result["type"])
	}
	if result["description"] != "A simple string" {
		t.Errorf("expected description 'A simple string', got %v", result["description"])
	}
}

func TestConvertSchema_WithEnum(t *testing.T) {
	schema := &genai.Schema{
		Type: "string",
		Enum: []string{"celsius", "fahrenheit"},
	}

	result := convertSchema(schema)

	enum, ok := result["enum"].([]string)
	if !ok {
		t.Fatalf("expected enum to be []string, got %T", result["enum"])
	}
	if len(enum) != 2 {
		t.Errorf("expected 2 enum values, got %d", len(enum))
	}
}

func TestConvertSchema_Object(t *testing.T) {
	schema := &genai.Schema{
		Type:        "object",
		Description: "Weather parameters",
		Properties: map[string]*genai.Schema{
			"city": {
				Type:        "string",
				Description: "City name",
			},
			"units": {
				Type: "string",
				Enum: []string{"celsius", "fahrenheit"},
			},
		},
		Required: []string{"city"},
	}

	result := convertSchema(schema)

	if result["type"] != "object" {
		t.Errorf("expected type 'object', got %v", result["type"])
	}

	props, ok := result["properties"].(map[string]any)
	if !ok {
		t.Fatalf("expected properties to be map[string]any, got %T", result["properties"])
	}
	if len(props) != 2 {
		t.Errorf("expected 2 properties, got %d", len(props))
	}

	required, ok := result["required"].([]string)
	if !ok {
		t.Fatalf("expected required to be []string, got %T", result["required"])
	}
	if len(required) != 1 || required[0] != "city" {
		t.Errorf("expected required ['city'], got %v", required)
	}
}

func TestConvertSchema_Array(t *testing.T) {
	schema := &genai.Schema{
		Type: "array",
		Items: &genai.Schema{
			Type: "string",
		},
	}

	result := convertSchema(schema)

	if result["type"] != "array" {
		t.Errorf("expected type 'array', got %v", result["type"])
	}

	items, ok := result["items"].(map[string]any)
	if !ok {
		t.Fatalf("expected items to be map[string]any, got %T", result["items"])
	}
	if items["type"] != "string" {
		t.Errorf("expected items type 'string', got %v", items["type"])
	}
}

func TestConvertSchema_Empty(t *testing.T) {
	schema := &genai.Schema{}

	result := convertSchema(schema)

	if len(result) != 0 {
		t.Errorf("expected empty map for empty schema, got %v", result)
	}
}

// ============================================================================
// convertFunctionDeclaration Tests
// ============================================================================

func TestConvertFunctionDeclaration(t *testing.T) {
	fn := &genai.FunctionDeclaration{
		Name:        "get_weather",
		Description: "Get weather for a city",
		Parameters: &genai.Schema{
			Type: "object",
			Properties: map[string]*genai.Schema{
				"city": {Type: "string"},
			},
			Required: []string{"city"},
		},
	}

	result := convertFunctionDeclaration(fn)

	if result.Type != openai.ToolTypeFunction {
		t.Errorf("expected type 'function', got %v", result.Type)
	}
	if result.Function.Name != "get_weather" {
		t.Errorf("expected name 'get_weather', got %v", result.Function.Name)
	}
	if result.Function.Description != "Get weather for a city" {
		t.Errorf("expected description 'Get weather for a city', got %v", result.Function.Description)
	}
	if result.Function.Parameters == nil {
		t.Error("expected parameters to be non-nil")
	}
}

func TestConvertFunctionDeclaration_WithJsonSchema(t *testing.T) {
	fn := &genai.FunctionDeclaration{
		Name:        "search",
		Description: "Search the web",
		ParametersJsonSchema: map[string]any{
			"type": "object",
			"properties": map[string]any{
				"query": map[string]any{"type": "string"},
			},
		},
	}

	result := convertFunctionDeclaration(fn)

	if result.Function.Parameters == nil {
		t.Error("expected parameters to be non-nil for JSON schema")
	}
}

func TestConvertFunctionDeclaration_NoParameters(t *testing.T) {
	fn := &genai.FunctionDeclaration{
		Name:        "get_time",
		Description: "Get current time",
	}

	result := convertFunctionDeclaration(fn)

	if result.Function.Parameters != nil {
		t.Errorf("expected nil parameters, got %v", result.Function.Parameters)
	}
}

// ============================================================================
// convertContent Tests
// ============================================================================

func TestConvertContent_TextMessage(t *testing.T) {
	m := &OpenRouterModel{}

	content := &genai.Content{
		Role: "user",
		Parts: []*genai.Part{
			genai.NewPartFromText("Hello, world!"),
		},
	}

	messages, err := m.convertContent(content)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}
	if messages[0].Role != openai.ChatMessageRoleUser {
		t.Errorf("expected role 'user', got %q", messages[0].Role)
	}
	if messages[0].Content != "Hello, world!" {
		t.Errorf("expected content 'Hello, world!', got %q", messages[0].Content)
	}
}

func TestConvertContent_ModelTextMessage(t *testing.T) {
	m := &OpenRouterModel{}

	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			genai.NewPartFromText("I am an AI assistant."),
		},
	}

	messages, err := m.convertContent(content)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}
	if messages[0].Role != openai.ChatMessageRoleAssistant {
		t.Errorf("expected role 'assistant', got %q", messages[0].Role)
	}
}

func TestConvertContent_FunctionCall(t *testing.T) {
	m := &OpenRouterModel{}

	content := &genai.Content{
		Role: "model",
		Parts: []*genai.Part{
			genai.NewPartFromFunctionCall("get_weather", map[string]any{
				"city": "London",
			}),
		},
	}
	// Set the ID on the function call
	content.Parts[0].FunctionCall.ID = "call_123"

	messages, err := m.convertContent(content)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}
	if len(messages[0].ToolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(messages[0].ToolCalls))
	}
	if messages[0].ToolCalls[0].Function.Name != "get_weather" {
		t.Errorf("expected function name 'get_weather', got %q", messages[0].ToolCalls[0].Function.Name)
	}
	if messages[0].ToolCalls[0].ID != "call_123" {
		t.Errorf("expected tool call ID 'call_123', got %q", messages[0].ToolCalls[0].ID)
	}
}


func TestConvertContent_FunctionResponse(t *testing.T) {
	m := &OpenRouterModel{}

	part := &genai.Part{
		FunctionResponse: &genai.FunctionResponse{
			ID:       "call_123",
			Name:     "get_weather",
			Response: map[string]any{"temperature": 20, "unit": "celsius"},
		},
	}
	content := &genai.Content{
		Role:  "tool",
		Parts: []*genai.Part{part},
	}

	messages, err := m.convertContent(content)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(messages))
	}
	if messages[0].Role != openai.ChatMessageRoleTool {
		t.Errorf("expected role 'tool', got %q", messages[0].Role)
	}
	if messages[0].ToolCallID != "call_123" {
		t.Errorf("expected tool call ID 'call_123', got %q", messages[0].ToolCallID)
	}
}

func TestConvertContent_EmptyContent(t *testing.T) {
	m := &OpenRouterModel{}

	content := &genai.Content{
		Role:  "user",
		Parts: []*genai.Part{},
	}

	messages, err := m.convertContent(content)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(messages) != 0 {
		t.Errorf("expected 0 messages for empty content, got %d", len(messages))
	}
}

// ============================================================================
// convertResponse Tests
// ============================================================================

func TestConvertResponse_TextOnly(t *testing.T) {
	m := &OpenRouterModel{}

	msg := &openai.ChatCompletionMessage{
		Role:    openai.ChatMessageRoleAssistant,
		Content: "Hello from the model!",
	}

	result := m.convertResponse(msg)

	if result.Content == nil {
		t.Fatal("expected content to be non-nil")
	}
	if result.Content.Role != "model" {
		t.Errorf("expected role 'model', got %q", result.Content.Role)
	}
	if len(result.Content.Parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(result.Content.Parts))
	}
	if result.Content.Parts[0].Text != "Hello from the model!" {
		t.Errorf("expected text 'Hello from the model!', got %q", result.Content.Parts[0].Text)
	}
}

func TestConvertResponse_WithToolCalls(t *testing.T) {
	m := &OpenRouterModel{}

	msg := &openai.ChatCompletionMessage{
		Role: openai.ChatMessageRoleAssistant,
		ToolCalls: []openai.ToolCall{
			{
				ID:   "call_abc123",
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      "get_weather",
					Arguments: `{"city":"Paris"}`,
				},
			},
		},
	}

	result := m.convertResponse(msg)

	if result.Content == nil {
		t.Fatal("expected content to be non-nil")
	}
	if len(result.Content.Parts) != 1 {
		t.Fatalf("expected 1 part, got %d", len(result.Content.Parts))
	}
	if result.Content.Parts[0].FunctionCall == nil {
		t.Fatal("expected function call to be non-nil")
	}
	if result.Content.Parts[0].FunctionCall.Name != "get_weather" {
		t.Errorf("expected function name 'get_weather', got %q", result.Content.Parts[0].FunctionCall.Name)
	}
	if result.Content.Parts[0].FunctionCall.ID != "call_abc123" {
		t.Errorf("expected function call ID 'call_abc123', got %q", result.Content.Parts[0].FunctionCall.ID)
	}
}

func TestConvertResponse_EmptyMessage(t *testing.T) {
	m := &OpenRouterModel{}

	msg := &openai.ChatCompletionMessage{
		Role: openai.ChatMessageRoleAssistant,
	}

	result := m.convertResponse(msg)

	if result.Content == nil {
		t.Fatal("expected content to be non-nil")
	}
	if len(result.Content.Parts) != 0 {
		t.Errorf("expected 0 parts for empty message, got %d", len(result.Content.Parts))
	}
}



// ============================================================================
// convertRequest Tests
// ============================================================================

func TestConvertRequest_Basic(t *testing.T) {
	m := &OpenRouterModel{modelName: "openai/gpt-4"}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					genai.NewPartFromText("Hello!"),
				},
			},
		},
	}

	result, err := m.convertRequest(req)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Model != "openai/gpt-4" {
		t.Errorf("expected model 'openai/gpt-4', got %q", result.Model)
	}
	if len(result.Messages) != 1 {
		t.Fatalf("expected 1 message, got %d", len(result.Messages))
	}
}

func TestConvertRequest_WithSystemInstruction(t *testing.T) {
	m := &OpenRouterModel{modelName: "test-model"}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					genai.NewPartFromText("What time is it?"),
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			SystemInstruction: &genai.Content{
				Parts: []*genai.Part{
					genai.NewPartFromText("You are a helpful assistant."),
				},
			},
		},
	}

	result, err := m.convertRequest(req)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// System message should be prepended
	if len(result.Messages) != 2 {
		t.Fatalf("expected 2 messages (system + user), got %d", len(result.Messages))
	}
	if result.Messages[0].Role != openai.ChatMessageRoleSystem {
		t.Errorf("expected first message to be system, got %q", result.Messages[0].Role)
	}
	if result.Messages[0].Content != "You are a helpful assistant." {
		t.Errorf("unexpected system message content: %q", result.Messages[0].Content)
	}
}

func TestConvertRequest_WithTools(t *testing.T) {
	m := &OpenRouterModel{modelName: "test-model"}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					genai.NewPartFromText("What's the weather?"),
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			Tools: []*genai.Tool{
				{
					FunctionDeclarations: []*genai.FunctionDeclaration{
						{
							Name:        "get_weather",
							Description: "Get weather for a city",
							Parameters: &genai.Schema{
								Type: "object",
								Properties: map[string]*genai.Schema{
									"city": {Type: "string"},
								},
							},
						},
					},
				},
			},
		},
	}

	result, err := m.convertRequest(req)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Tools) != 1 {
		t.Fatalf("expected 1 tool, got %d", len(result.Tools))
	}
	if result.Tools[0].Function.Name != "get_weather" {
		t.Errorf("expected tool name 'get_weather', got %q", result.Tools[0].Function.Name)
	}
}

func TestConvertRequest_WithGenerationConfig(t *testing.T) {
	m := &OpenRouterModel{modelName: "test-model"}

	temp := float32(0.7)
	topP := float32(0.9)

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role: "user",
				Parts: []*genai.Part{
					genai.NewPartFromText("Hello!"),
				},
			},
		},
		Config: &genai.GenerateContentConfig{
			Temperature:     &temp,
			TopP:            &topP,
			MaxOutputTokens: 1000,
			StopSequences:   []string{"END", "STOP"},
		},
	}

	result, err := m.convertRequest(req)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if result.Temperature != 0.7 {
		t.Errorf("expected temperature 0.7, got %v", result.Temperature)
	}
	if result.TopP != 0.9 {
		t.Errorf("expected top_p 0.9, got %v", result.TopP)
	}
	if result.MaxCompletionTokens != 1000 {
		t.Errorf("expected max_tokens 1000, got %d", result.MaxCompletionTokens)
	}
	if len(result.Stop) != 2 {
		t.Errorf("expected 2 stop sequences, got %d", len(result.Stop))
	}
}

func TestConvertRequest_EmptyContents(t *testing.T) {
	m := &OpenRouterModel{modelName: "test-model"}

	req := &model.LLMRequest{
		Contents: []*genai.Content{},
	}

	result, err := m.convertRequest(req)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Messages) != 0 {
		t.Errorf("expected 0 messages, got %d", len(result.Messages))
	}
}

func TestConvertRequest_MultipleMessages(t *testing.T) {
	m := &OpenRouterModel{modelName: "test-model"}

	req := &model.LLMRequest{
		Contents: []*genai.Content{
			{
				Role:  "user",
				Parts: []*genai.Part{genai.NewPartFromText("Hello")},
			},
			{
				Role:  "model",
				Parts: []*genai.Part{genai.NewPartFromText("Hi there!")},
			},
			{
				Role:  "user",
				Parts: []*genai.Part{genai.NewPartFromText("How are you?")},
			},
		},
	}

	result, err := m.convertRequest(req)

	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(result.Messages) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(result.Messages))
	}
	if result.Messages[0].Role != openai.ChatMessageRoleUser {
		t.Errorf("expected first message role 'user', got %q", result.Messages[0].Role)
	}
	if result.Messages[1].Role != openai.ChatMessageRoleAssistant {
		t.Errorf("expected second message role 'assistant', got %q", result.Messages[1].Role)
	}
	if result.Messages[2].Role != openai.ChatMessageRoleUser {
		t.Errorf("expected third message role 'user', got %q", result.Messages[2].Role)
	}
}