// enc_check — the engine's V4.1 chat encoding for a few fixed conversations, one per line (escaped),
// to diff against DeepSeek-V4.1's encoding.py (tools/v41/enc_check.py prints the reference side).
#include "engine/chat_encode.h"
#include <cstdio>
#include <string>
#include <vector>
using namespace aff;
static void emit(const std::vector<ChatMsg>& m, ThinkingMode t, int b) {
  EncodeOpts o; o.thinking_mode = t; o.v41 = true; o.effort_budget = b;
  const std::string s = encode_messages(m, o);
  for (char c : s) { if (c == '\n') std::fputs("\\n", stdout); else std::fputc(c, stdout); }
  std::fputc('\n', stdout);
}
int main() {
  ChatMsg u; u.role = "user"; u.content = "What is the capital of France?";
  ChatMsg sy; sy.role = "system"; sy.content = "You are terse.";
  ChatMsg a; a.role = "assistant"; a.content = "Paris."; a.reasoning_content = "Easy.";
  ChatMsg u2; u2.role = "user"; u2.content = "And Spain?";
  emit({u}, ThinkingMode::Chat, 100);
  emit({u}, ThinkingMode::Thinking, 100);
  emit({u}, ThinkingMode::Thinking, 75);
  emit({sy, u}, ThinkingMode::Chat, 100);
  emit({sy, u}, ThinkingMode::Thinking, 50);
  emit({u, a, u2}, ThinkingMode::Thinking, 100);
  emit({u, a, sy, u2}, ThinkingMode::Chat, 100);
}
