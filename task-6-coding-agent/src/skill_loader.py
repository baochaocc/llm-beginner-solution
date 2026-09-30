"""任务六 M2：Skill 加载器（≈50 行）——渐进式披露。

扫描 `skills_dir/*/SKILL.md`，用 YAML front-matter 元数据（name/description）建索引；
`list_skills()` 只给元数据，`load(name)` 命中后才把完整正文拿出来（progressive
disclosure：先把全部 skill 正文塞进 context 就违背了这个设计）。

对齐 Anthropic Skills 约定：每个 skill 一个目录，SKILL.md 带 `---` front-matter。
"""
import re
from pathlib import Path

import yaml

_FRONT_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n?(.*)\Z", re.S)


class SkillLoader:
    def __init__(self, skills_dir: str):
        self.skills_dir = Path(skills_dir)
        self._skills = self._scan()

    def _scan(self) -> dict:
        skills = {}
        if not self.skills_dir.is_dir():
            return skills
        for md in sorted(self.skills_dir.glob("*/SKILL.md")):
            text = md.read_text(encoding="utf-8")
            m = _FRONT_RE.match(text)
            if not m:
                continue                      # 缺 front-matter 的忽略（自检要求每项都有元数据）
            try:
                meta = yaml.safe_load(m.group(1)) or {}
            except yaml.YAMLError:
                continue
            name = str(meta.get("name") or "").strip()
            if not name or name in skills:
                continue
            skills[name] = {
                "name": name,
                "path": str(md.relative_to(self.skills_dir.parent.parent)),  # src/skills/...
                "description": str(meta.get("description") or "").strip(),
                "body": m.group(2).strip(),
            }
        return skills

    def list_skills(self) -> list:
        """返回元数据列表（每项含 name + description）；不含正文。"""
        return [{k: v for k, v in s.items() if k != "body"} for s in self._skills.values()]

    def load(self, name: str) -> str:
        """按 name 加载完整 SKILL.md 正文（含 front-matter 元数据头）。"""
        if name not in self._skills:
            raise KeyError(f"未找到 Skill: {name}（可用：{list(self._skills)}）")
        s = self._skills[name]
        return f"# Skill: {s['name']}\n\n描述：{s['description']}\n\n---\n\n{s['body']}"
