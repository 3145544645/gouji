msg=$(cat)
if printf '%s' "$msg" | grep -qiE '^AI-Provider:'; then
  printf '%s' "$msg"
else
  printf 'AI-Provider: Claude\n\n%s' "$msg"
fi
