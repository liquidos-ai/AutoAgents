use crate::agent::skill::SkillSnapshot;
use std::collections::BTreeSet;
use std::fmt::Write;

pub(crate) struct SkillPromptRenderer;

impl SkillPromptRenderer {
    pub(crate) fn render(
        base_prompt: &str,
        snapshot: &SkillSnapshot,
        active_skills: &BTreeSet<String>,
    ) -> String {
        if snapshot.is_empty() {
            return base_prompt.to_string();
        }

        let catalog_bytes = snapshot
            .skills()
            .map(|skill| {
                skill.metadata().name().len()
                    + skill.metadata().description().len()
                    + skill.metadata().compatibility().map_or(0, str::len)
            })
            .sum::<usize>();
        let active_bytes = active_skills
            .iter()
            .filter_map(|name| snapshot.get_ref(name))
            .map(|skill| {
                skill.instructions().len()
                    + skill
                        .resources()
                        .iter()
                        .map(|resource| resource.as_os_str().as_encoded_bytes().len())
                        .sum::<usize>()
            })
            .sum::<usize>();
        let capacity = base_prompt
            .len()
            .saturating_add(catalog_bytes)
            .saturating_add(active_bytes)
            .saturating_add(snapshot.len().saturating_mul(64))
            .saturating_add(512);
        let mut prompt = String::with_capacity(capacity);
        prompt.push_str(base_prompt);
        prompt.push_str(
            "\n\n# Agent Skills\n\
             The skills below contain specialized instructions. When a task matches a skill \
             description, call `activate_skill` with only that skill name before taking other \
             actions. Do not call other tools in the same turn as `activate_skill`. Activated \
             instructions remain authoritative for this conversation until `deactivate_skill` is \
             called or the skill is removed. Use `read_skill_resource` only for resources belonging \
             to an activated skill.\n\n\
             ## Available skills\n",
        );
        for skill in snapshot.skills() {
            prompt.push_str("- `");
            prompt.push_str(skill.metadata().name());
            prompt.push_str("`: ");
            prompt.push_str(skill.metadata().description());
            if let Some(compatibility) = skill.metadata().compatibility() {
                prompt.push_str(" Compatibility: ");
                prompt.push_str(compatibility);
            }
            prompt.push('\n');
        }

        for name in active_skills {
            let Some(skill) = snapshot.get_ref(name) else {
                continue;
            };
            prompt.push_str("\n## Active skill: `");
            prompt.push_str(skill.metadata().name());
            prompt.push_str("`\n\n");
            prompt.push_str(skill.instructions());
            prompt.push('\n');
            if !skill.resources().is_empty() {
                prompt.push_str("\nResources available through `read_skill_resource`:\n");
                for resource in skill.resources() {
                    prompt.push_str("- `");
                    let _ = write!(prompt, "{}", resource.display());
                    prompt.push_str("`\n");
                }
            }
        }
        prompt
    }
}
