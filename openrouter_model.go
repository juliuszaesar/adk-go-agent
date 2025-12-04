package main

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"iter"

	"github.com/sashabaranov/go-openai"
	"google.golang.org/adk/model"
	"google.golang.org/genai"
)

// OpenRouterModel implements the google.golang.org/adk/model.LLM interface
// for use with OpenRouter's OpenAI-compatible API.
type OpenRouterModel struct {
	client    *openai.Client
	modelName string
}

// OpenRouterConfig holds configuration options for the OpenRouter model.
type OpenRouterConfig struct {
	// APIKey is the OpenRouter API key (required)
	APIKey string
	// BaseURL is the OpenRouter API base URL (defaults to https://openrouter.ai/api/v1)
	BaseURL string
}

// NewOpenRouterModel creates a new OpenRouter model instance.
// modelName should be in OpenRouter format, e.g., "openai/gpt-4", "anthropic/claude-3-opus"
func NewOpenRouterModel(modelName string, cfg *OpenRouterConfig) (*OpenRouterModel, error) {
	if cfg == nil || cfg.APIKey == "" {
		return nil, fmt.Errorf("OpenRouter API key is required")
	}

	config := openai.DefaultConfig(cfg.APIKey)
	if cfg.BaseURL != "" {
		config.BaseURL = cfg.BaseURL
	} else {
		config.BaseURL = "https://openrouter.ai/api/v1"
	}

	return &OpenRouterModel{
		client:    openai.NewClientWithConfig(config),
		modelName: modelName,
	}, nil
}

// Name returns the model name.
func (m *OpenRouterModel) Name() string {
	return m.modelName
}

// GenerateContent implements the model.LLM interface.
// It converts ADK requests to OpenAI format, calls OpenRouter, and converts responses back.
func (m *OpenRouterModel) GenerateContent(ctx context.Context, req *model.LLMRequest, stream bool) iter.Seq2[*model.LLMResponse, error] {
	return func(yield func(*model.LLMResponse, error) bool) {
		// Convert ADK request to OpenAI format
		openaiReq, err := m.convertRequest(req)
		if err != nil {
			yield(nil, fmt.Errorf("failed to convert request: %w", err))
			return
		}

		if stream {
			m.handleStreamingResponse(ctx, openaiReq, yield)
		} else {
			m.handleNonStreamingResponse(ctx, openaiReq, yield)
		}
	}
}

// convertRequest converts an ADK LLMRequest to an OpenAI ChatCompletionRequest.
func (m *OpenRouterModel) convertRequest(req *model.LLMRequest) (openai.ChatCompletionRequest, error) {
	openaiReq := openai.ChatCompletionRequest{
		Model: m.modelName,
	}

	// Convert messages
	for _, content := range req.Contents {
		msg, err := m.convertContent(content)
		if err != nil {
			return openaiReq, err
		}
		openaiReq.Messages = append(openaiReq.Messages, msg...)
	}

	// Convert system instruction if present
	if req.Config != nil && req.Config.SystemInstruction != nil {
		sysMsg := openai.ChatCompletionMessage{
			Role:    openai.ChatMessageRoleSystem,
			Content: extractText(req.Config.SystemInstruction),
		}
		// Prepend system message
		openaiReq.Messages = append([]openai.ChatCompletionMessage{sysMsg}, openaiReq.Messages...)
	}

	// Convert tools from Config
	if req.Config != nil && len(req.Config.Tools) > 0 {
		for _, tool := range req.Config.Tools {
			if tool.FunctionDeclarations != nil {
				for _, fn := range tool.FunctionDeclarations {
					openaiReq.Tools = append(openaiReq.Tools, convertFunctionDeclaration(fn))
				}
			}
		}
	}

	// Apply generation config
	if req.Config != nil {
		if req.Config.Temperature != nil {
			openaiReq.Temperature = *req.Config.Temperature
		}
		if req.Config.TopP != nil {
			openaiReq.TopP = *req.Config.TopP
		}
		if req.Config.MaxOutputTokens > 0 {
			openaiReq.MaxCompletionTokens = int(req.Config.MaxOutputTokens)
		}
		if len(req.Config.StopSequences) > 0 {
			openaiReq.Stop = req.Config.StopSequences
		}
	}

	return openaiReq, nil
}

// convertContent converts a genai.Content to OpenAI ChatCompletionMessage(s).
func (m *OpenRouterModel) convertContent(content *genai.Content) ([]openai.ChatCompletionMessage, error) {
	var messages []openai.ChatCompletionMessage

	role := convertRole(content.Role)

	// Check if this content contains function calls or function responses
	var textParts []string
	var toolCalls []openai.ToolCall

	for _, part := range content.Parts {
		if part.Text != "" {
			textParts = append(textParts, part.Text)
		}
		if part.FunctionCall != nil {
			// Model is requesting a function call
			argsJSON, err := json.Marshal(part.FunctionCall.Args)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function call args: %w", err)
			}
			toolCalls = append(toolCalls, openai.ToolCall{
				ID:   part.FunctionCall.ID,
				Type: openai.ToolTypeFunction,
				Function: openai.FunctionCall{
					Name:      part.FunctionCall.Name,
					Arguments: string(argsJSON),
				},
			})
		}
		if part.FunctionResponse != nil {
			// This is a tool response - needs special handling
			responseJSON, err := json.Marshal(part.FunctionResponse.Response)
			if err != nil {
				return nil, fmt.Errorf("failed to marshal function response: %w", err)
			}
			messages = append(messages, openai.ChatCompletionMessage{
				Role:       openai.ChatMessageRoleTool,
				Content:    string(responseJSON),
				ToolCallID: part.FunctionResponse.ID,
			})
		}
	}

	// If we have text or tool calls, create a message
	if len(textParts) > 0 || len(toolCalls) > 0 {
		msg := openai.ChatCompletionMessage{
			Role: role,
		}
		if len(textParts) > 0 {
			msg.Content = joinStrings(textParts)
		}
		if len(toolCalls) > 0 {
			msg.ToolCalls = toolCalls
		}
		messages = append(messages, msg)
	}

	return messages, nil
}

// handleNonStreamingResponse handles non-streaming API calls.
func (m *OpenRouterModel) handleNonStreamingResponse(ctx context.Context, req openai.ChatCompletionRequest, yield func(*model.LLMResponse, error) bool) {
	resp, err := m.client.CreateChatCompletion(ctx, req)
	if err != nil {
		yield(nil, fmt.Errorf("openrouter error: %w", err))
		return
	}

	if len(resp.Choices) == 0 {
		yield(nil, fmt.Errorf("openrouter returned no choices"))
		return
	}

	choice := resp.Choices[0]
	llmResp := m.convertResponse(&choice.Message)
	llmResp.TurnComplete = true
	llmResp.FinishReason = convertFinishReason(choice.FinishReason)

	// Add usage metadata if available
	if resp.Usage.TotalTokens > 0 {
		llmResp.UsageMetadata = &genai.GenerateContentResponseUsageMetadata{
			PromptTokenCount:     int32(resp.Usage.PromptTokens),
			CandidatesTokenCount: int32(resp.Usage.CompletionTokens),
			TotalTokenCount:      int32(resp.Usage.TotalTokens),
		}
	}

	yield(llmResp, nil)
}


// handleStreamingResponse handles streaming API calls.
func (m *OpenRouterModel) handleStreamingResponse(ctx context.Context, req openai.ChatCompletionRequest, yield func(*model.LLMResponse, error) bool) {
	req.Stream = true

	stream, err := m.client.CreateChatCompletionStream(ctx, req)
	if err != nil {
		yield(nil, fmt.Errorf("openrouter stream error: %w", err))
		return
	}
	defer stream.Close()

	var accumulatedContent string
	var accumulatedToolCalls []openai.ToolCall

	for {
		chunk, err := stream.Recv()
		if err == io.EOF {
			break
		}
		if err != nil {
			yield(nil, fmt.Errorf("openrouter stream recv error: %w", err))
			return
		}

		if len(chunk.Choices) == 0 {
			continue
		}

		delta := chunk.Choices[0].Delta
		finishReason := chunk.Choices[0].FinishReason

		// Accumulate content
		if delta.Content != "" {
			accumulatedContent += delta.Content

			// Yield partial response for text streaming
			llmResp := &model.LLMResponse{
				Content: genai.NewContentFromText(delta.Content, "model"),
				Partial: true,
			}
			if !yield(llmResp, nil) {
				return
			}
		}

		// Accumulate tool calls
		for _, tc := range delta.ToolCalls {
			if tc.Index != nil {
				idx := *tc.Index
				// Extend slice if needed
				for len(accumulatedToolCalls) <= idx {
					accumulatedToolCalls = append(accumulatedToolCalls, openai.ToolCall{})
				}
				// Merge tool call data
				if tc.ID != "" {
					accumulatedToolCalls[idx].ID = tc.ID
				}
				if tc.Type != "" {
					accumulatedToolCalls[idx].Type = tc.Type
				}
				if tc.Function.Name != "" {
					accumulatedToolCalls[idx].Function.Name = tc.Function.Name
				}
				accumulatedToolCalls[idx].Function.Arguments += tc.Function.Arguments
			}
		}

		// Check if stream is complete
		if finishReason != "" {
			// Build final response
			finalMsg := openai.ChatCompletionMessage{
				Role:      openai.ChatMessageRoleAssistant,
				Content:   accumulatedContent,
				ToolCalls: accumulatedToolCalls,
			}

			llmResp := m.convertResponse(&finalMsg)
			llmResp.TurnComplete = true
			llmResp.Partial = false
			llmResp.FinishReason = convertFinishReason(finishReason)

			yield(llmResp, nil)
			return
		}
	}
}

// convertResponse converts an OpenAI ChatCompletionMessage to an ADK LLMResponse.
func (m *OpenRouterModel) convertResponse(msg *openai.ChatCompletionMessage) *model.LLMResponse {
	var parts []*genai.Part

	// Add text content
	if msg.Content != "" {
		parts = append(parts, genai.NewPartFromText(msg.Content))
	}

	// Add function calls
	for _, tc := range msg.ToolCalls {
		if tc.Type == openai.ToolTypeFunction {
			var args map[string]any
			if tc.Function.Arguments != "" {
				_ = json.Unmarshal([]byte(tc.Function.Arguments), &args)
			}
			parts = append(parts, genai.NewPartFromFunctionCall(tc.Function.Name, args))
			// Set the ID on the function call
			if len(parts) > 0 && parts[len(parts)-1].FunctionCall != nil {
				parts[len(parts)-1].FunctionCall.ID = tc.ID
			}
		}
	}

	content := &genai.Content{
		Role:  "model",
		Parts: parts,
	}

	return &model.LLMResponse{
		Content: content,
	}
}

// Helper functions

// convertRole converts ADK role to OpenAI role.
func convertRole(role string) string {
	switch role {
	case "model", "assistant":
		return openai.ChatMessageRoleAssistant
	case "system":
		return openai.ChatMessageRoleSystem
	case "tool":
		return openai.ChatMessageRoleTool
	default:
		return openai.ChatMessageRoleUser
	}
}

// convertFunctionDeclaration converts a genai.FunctionDeclaration to an OpenAI Tool.
func convertFunctionDeclaration(fn *genai.FunctionDeclaration) openai.Tool {
	var params any
	if fn.Parameters != nil {
		params = convertSchema(fn.Parameters)
	} else if fn.ParametersJsonSchema != nil {
		params = fn.ParametersJsonSchema
	}

	return openai.Tool{
		Type: openai.ToolTypeFunction,
		Function: &openai.FunctionDefinition{
			Name:        fn.Name,
			Description: fn.Description,
			Parameters:  params,
		},
	}
}

// convertSchema converts a genai.Schema to a map for OpenAI.
func convertSchema(schema *genai.Schema) map[string]any {
	result := make(map[string]any)

	if schema.Type != "" {
		result["type"] = string(schema.Type)
	}
	if schema.Description != "" {
		result["description"] = schema.Description
	}
	if len(schema.Enum) > 0 {
		result["enum"] = schema.Enum
	}
	if schema.Items != nil {
		result["items"] = convertSchema(schema.Items)
	}
	if len(schema.Properties) > 0 {
		props := make(map[string]any)
		for name, prop := range schema.Properties {
			props[name] = convertSchema(prop)
		}
		result["properties"] = props
	}
	if len(schema.Required) > 0 {
		result["required"] = schema.Required
	}

	return result
}

// convertFinishReason converts OpenAI finish reason to genai.FinishReason.
func convertFinishReason(reason openai.FinishReason) genai.FinishReason {
	switch reason {
	case openai.FinishReasonStop:
		return genai.FinishReasonStop
	case openai.FinishReasonLength:
		return genai.FinishReasonMaxTokens
	case openai.FinishReasonToolCalls, openai.FinishReasonFunctionCall:
		return genai.FinishReasonStop // Tool calls are considered a valid stop
	default:
		return genai.FinishReasonUnspecified
	}
}

// extractText extracts all text from a genai.Content.
func extractText(content *genai.Content) string {
	var texts []string
	for _, part := range content.Parts {
		if part.Text != "" {
			texts = append(texts, part.Text)
		}
	}
	return joinStrings(texts)
}

// joinStrings joins strings with no separator.
func joinStrings(strs []string) string {
	result := ""
	for _, s := range strs {
		result += s
	}
	return result
}
