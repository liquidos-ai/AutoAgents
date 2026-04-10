use autoagents::async_trait;
use autoagents::core::tool::{ToolCallError, ToolRuntime, ToolT};
use autoagents_derive::{ToolInput, tool};
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::fs;
use std::path::Path;
use walkdir::WalkDir;

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct GrepArgs {
    #[input(description = "Regular expression pattern to search for")]
    pattern: String,
    #[input(description = "File glob pattern to search in (e.g., '*.rs')")]
    file_pattern: String,
    #[input(description = "Base directory to search in")]
    base_dir: String,
}

#[tool(
    name = "GrepTool",
    description = "Search for content in files using regex patterns",
    input = GrepArgs,
)]
pub struct GrepTool {}

#[async_trait]
impl ToolRuntime for GrepTool {
    async fn execute(&self, args: serde_json::Value) -> Result<serde_json::Value, ToolCallError> {
        let args: GrepArgs = serde_json::from_value(args)?;
        println!("🔎 Grepping for: {} in {}", args.pattern, args.file_pattern);

        let regex = Regex::new(&args.pattern)
            .map_err(|e| ToolCallError::RuntimeError(format!("Invalid regex: {}", e).into()))?;

        let base_path = Path::new(&args.base_dir);
        if !base_path.exists() {
            return Err(ToolCallError::RuntimeError(
                format!("Directory {} does not exist", args.base_dir).into(),
            ));
        }

        let file_pattern = glob::Pattern::new(&args.file_pattern).map_err(|e| {
            ToolCallError::RuntimeError(format!("Invalid file pattern: {}", e).into())
        })?;

        let mut results = Vec::new();
        let max_results = 50;

        for entry in WalkDir::new(&args.base_dir)
            .follow_links(true)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if results.len() >= max_results {
                break;
            }

            let path = entry.path();
            if path.is_file() {
                let relative_path = path.strip_prefix(&args.base_dir).unwrap_or(path);
                if file_pattern.matches_path(relative_path)
                    && let Ok(content) = fs::read_to_string(path)
                {
                    for (line_num, line) in content.lines().enumerate() {
                        if regex.is_match(line) {
                            results.push(format!(
                                "{}:{}: {}",
                                relative_path.display(),
                                line_num + 1,
                                line.trim()
                            ));
                            if results.len() >= max_results {
                                break;
                            }
                        }
                    }
                }
            }
        }

        if results.is_empty() {
            Ok("No matches found.".to_string().into())
        } else {
            Ok(format!(
                "Found {} matches (showing up to {}):\n{}",
                results.len(),
                max_results,
                results.join("\n")
            )
            .into())
        }
    }
}

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct AnalyzeCodeArgs {
    #[input(description = "Path to the file or directory to analyze")]
    path: String,
    #[input(description = "Type of analysis: 'structure', 'complexity', 'dependencies'")]
    analysis_type: String,
}

#[tool(
    name = "AnalyzeCodeTool",
    description = "Analyze code structure, complexity, or dependencies",
    input = AnalyzeCodeArgs,
)]
pub struct AnalyzeCodeTool {}

#[async_trait]
impl ToolRuntime for AnalyzeCodeTool {
    async fn execute(&self, args: serde_json::Value) -> Result<serde_json::Value, ToolCallError> {
        let args: AnalyzeCodeArgs = serde_json::from_value(args)?;
        println!("🔬 Analyzing code: {} ({})", args.path, args.analysis_type);

        let path = Path::new(&args.path);
        if !path.exists() {
            return Err(ToolCallError::RuntimeError(
                format!("Path {} does not exist", args.path).into(),
            ));
        }

        match args.analysis_type.as_str() {
            "structure" => Ok(analyze_structure(path)?.into()),
            "complexity" => Ok(analyze_complexity(path)?.into()),
            "dependencies" => Ok(analyze_dependencies(path)?.into()),
            _ => Err(ToolCallError::RuntimeError(
                "Invalid analysis type. Choose 'structure', 'complexity', or 'dependencies'".into(),
            )),
        }
    }
}

fn analyze_structure(path: &Path) -> Result<String, ToolCallError> {
    let mut file_count = 0;
    let mut dir_count = 0;
    let mut total_lines = 0;
    let mut extensions: std::collections::HashMap<String, usize> = std::collections::HashMap::new();

    if path.is_file() {
        file_count = 1;
        if let Ok(content) = fs::read_to_string(path) {
            total_lines = content.lines().count();
        }
        if let Some(ext) = path.extension() {
            *extensions
                .entry(ext.to_string_lossy().to_string())
                .or_insert(0) += 1;
        }
    } else {
        for entry in WalkDir::new(path).into_iter().filter_map(|e| e.ok()) {
            let entry_path = entry.path();
            if entry_path.is_file() {
                file_count += 1;
                if let Ok(content) = fs::read_to_string(entry_path) {
                    total_lines += content.lines().count();
                }
                if let Some(ext) = entry_path.extension() {
                    *extensions
                        .entry(ext.to_string_lossy().to_string())
                        .or_insert(0) += 1;
                }
            } else if entry_path.is_dir() {
                dir_count += 1;
            }
        }
    }

    let mut ext_summary = String::new();
    for (ext, count) in extensions.iter() {
        ext_summary.push_str(&format!("\n  .{}: {} files", ext, count));
    }

    Ok(format!(
        "Code Structure Analysis:\n\
        - Files: {}\n\
        - Directories: {}\n\
        - Total lines: {}\n\
        - File types:{}",
        file_count, dir_count, total_lines, ext_summary
    ))
}

fn analyze_complexity(_path: &Path) -> Result<String, ToolCallError> {
    // Simplified complexity analysis
    Ok(
        "Complexity analysis: This is a placeholder. In a real implementation, \
        this would calculate cyclomatic complexity, function lengths, and other metrics."
            .to_string(),
    )
}

fn analyze_dependencies(_path: &Path) -> Result<String, ToolCallError> {
    // Simplified dependency analysis
    Ok(
        "Dependency analysis: This is a placeholder. In a real implementation, \
        this would parse import statements and analyze module dependencies."
            .to_string(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::path::{Path, PathBuf};
    use std::time::{SystemTime, UNIX_EPOCH};

    fn temp_fixture_dir(label: &str) -> PathBuf {
        let unique = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .expect("system clock should be after unix epoch")
            .as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "autoagents-coding-agent-{label}-{}-{unique}",
            std::process::id()
        ));
        fs::create_dir_all(&dir).expect("temp dir should create");
        dir
    }

    fn write(path: &Path, content: &str) {
        if let Some(parent) = path.parent() {
            fs::create_dir_all(parent).expect("parent directory should create");
        }
        fs::write(path, content).expect("fixture file should write");
    }

    #[tokio::test]
    async fn grep_tool_finds_matches_with_expected_glob_filter() {
        let dir = temp_fixture_dir("grep");
        write(
            &dir.join("src/lib.rs"),
            "fn needle() {}\nlet ignored = 1;\n",
        );
        write(
            &dir.join("README.md"),
            "needle should not match this glob\n",
        );

        let result = GrepTool {}
            .execute(json!({
                "pattern": "needle",
                "file_pattern": "src/*.rs",
                "base_dir": dir.to_string_lossy(),
            }))
            .await
            .expect("grep should succeed");

        let output = result.as_str().expect("grep output should be a string");
        assert!(output.contains("Found 1 matches"));
        assert!(output.contains("src/lib.rs:1: fn needle() {}"));
        assert!(!output.contains("README.md"));

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn grep_tool_reports_invalid_inputs_and_no_matches() {
        let dir = temp_fixture_dir("grep-errors");
        write(&dir.join("src/lib.rs"), "fn present() {}\n");

        let no_match = GrepTool {}
            .execute(json!({
                "pattern": "missing",
                "file_pattern": "src/*.rs",
                "base_dir": dir.to_string_lossy(),
            }))
            .await
            .expect("grep should succeed even when there are no matches");
        assert_eq!(no_match, json!("No matches found."));

        let invalid_regex = GrepTool {}
            .execute(json!({
                "pattern": "(",
                "file_pattern": "src/*.rs",
                "base_dir": dir.to_string_lossy(),
            }))
            .await
            .expect_err("invalid regex should fail");
        assert!(invalid_regex.to_string().contains("Invalid regex"));

        let invalid_glob = GrepTool {}
            .execute(json!({
                "pattern": "present",
                "file_pattern": "[*.rs",
                "base_dir": dir.to_string_lossy(),
            }))
            .await
            .expect_err("invalid glob should fail");
        assert!(invalid_glob.to_string().contains("Invalid file pattern"));

        let missing_dir = GrepTool {}
            .execute(json!({
                "pattern": "present",
                "file_pattern": "*.rs",
                "base_dir": dir.join("missing").to_string_lossy(),
            }))
            .await
            .expect_err("missing directory should fail");
        assert!(missing_dir.to_string().contains("does not exist"));

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn analyze_code_tool_covers_all_supported_analysis_modes() {
        let dir = temp_fixture_dir("analyze");
        write(
            &dir.join("src/lib.rs"),
            "pub fn add(a: i32, b: i32) -> i32 {\n    a + b\n}\n",
        );
        write(
            &dir.join("src/nested/mod.py"),
            "def greet(name):\n    return name\n",
        );

        let structure = AnalyzeCodeTool {}
            .execute(json!({
                "path": dir.to_string_lossy(),
                "analysis_type": "structure",
            }))
            .await
            .expect("structure analysis should succeed");
        let summary = structure.as_str().expect("summary should be a string");
        assert!(summary.contains("Code Structure Analysis"));
        assert!(summary.contains("- Files: 2"));
        assert!(summary.contains(".rs: 1 files"));
        assert!(summary.contains(".py: 1 files"));

        let single_file_summary = analyze_structure(&dir.join("src/lib.rs"))
            .expect("single-file structure analysis should succeed");
        assert!(single_file_summary.contains("- Files: 1"));
        assert!(single_file_summary.contains(".rs: 1 files"));

        let complexity = AnalyzeCodeTool {}
            .execute(json!({
                "path": dir.to_string_lossy(),
                "analysis_type": "complexity",
            }))
            .await
            .expect("complexity analysis should succeed");
        assert!(
            complexity
                .as_str()
                .expect("complexity output should be a string")
                .contains("cyclomatic complexity")
        );

        let dependencies = AnalyzeCodeTool {}
            .execute(json!({
                "path": dir.to_string_lossy(),
                "analysis_type": "dependencies",
            }))
            .await
            .expect("dependency analysis should succeed");
        assert!(
            dependencies
                .as_str()
                .expect("dependency output should be a string")
                .contains("Dependency analysis")
        );

        let invalid_type = AnalyzeCodeTool {}
            .execute(json!({
                "path": dir.to_string_lossy(),
                "analysis_type": "unknown",
            }))
            .await
            .expect_err("unknown analysis type should fail");
        assert!(invalid_type.to_string().contains("Invalid analysis type"));

        let missing_path = AnalyzeCodeTool {}
            .execute(json!({
                "path": dir.join("missing").to_string_lossy(),
                "analysis_type": "structure",
            }))
            .await
            .expect_err("missing path should fail");
        assert!(missing_path.to_string().contains("does not exist"));

        let _ = fs::remove_dir_all(dir);
    }
}
