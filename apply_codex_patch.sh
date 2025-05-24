#!/bin/zsh

echo "⚠️ Codex のパッチをペーストして Ctrl-D を押してください"
cat > fix_codex.patch
echo "✅ パッチ保存完了: fix_codex.patch"

git apply --3way fix_codex.patch || { echo "❌ git apply 失敗"; exit 1; }

git add .
git commit -m "Codex パッチ適用"

BRANCH=$(git branch --show-current)
echo "➡️ 現在のブランチ: $BRANCH"

git push origin "$BRANCH"
