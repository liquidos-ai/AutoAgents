use std::fs;
use std::path::Path;

use autoagents::async_trait;
use autoagents::core::tool::{ToolCallError, ToolRuntime, ToolT};
use autoagents_derive::{ToolInput, tool};
use autoagents_toolkit::tools::filesystem::FilesystemSandbox;
use glob::Pattern;
use regex::Regex;
use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct GrepArgs {
    #[input(description = "Regular expression pattern to search for")]
    pattern: String,
    #[input(description = "File glob pattern to search in (e.g., '*.rs')")]
    file_pattern: String,
    #[serde(default = "default_subdirectory")]
    #[input(description = "Relative subdirectory within the workspace to search in")]
    subdirectory: String,
}

fn default_subdirectory() -> String {
    ".".to_string()
}

const MAX_MATCH_RESULTS: usize = 50;
const MAX_FILES_SCANNED: usize = 500;

fn io_sandbox_error(error: std::io::Error) -> ToolCallError {
    ToolCallError::RuntimeError(Box::new(error))
}

#[tool(
    name = "GrepTool",
    description = "Search for content in files using regex patterns within the workspace sandbox",
    input = GrepArgs,
)]
pub struct GrepTool {
    sandbox: FilesystemSandbox,
}

impl GrepTool {
    pub fn with_sandbox(sandbox: FilesystemSandbox) -> Self {
        Self { sandbox }
    }
}

#[async_trait]
impl ToolRuntime for GrepTool {
    async fn execute(&self, args: serde_json::Value) -> Result<serde_json::Value, ToolCallError> {
        let args: GrepArgs = serde_json::from_value(args)?;

        let regex = Regex::new(&args.pattern)
            .map_err(|e| ToolCallError::RuntimeError(format!("Invalid regex: {}", e).into()))?;

        let base_path = self
            .sandbox
            .resolve_relative(&args.subdirectory)
            .map_err(io_sandbox_error)?;
        let base_path = self
            .sandbox
            .ensure_resolved(&base_path)
            .map_err(io_sandbox_error)?;

        if !base_path.exists() {
            return Err(ToolCallError::RuntimeError(
                format!("Subdirectory {} does not exist", base_path.display()).into(),
            ));
        }

        if !base_path.is_dir() {
            return Err(ToolCallError::RuntimeError(
                format!("Subdirectory {} is not a directory", base_path.display()).into(),
            ));
        }

        let file_pattern = Pattern::new(&args.file_pattern).map_err(|e| {
            ToolCallError::RuntimeError(format!("Invalid file pattern: {}", e).into())
        })?;

        let mut results = Vec::new();
        let mut files_scanned = 0usize;

        for entry in self
            .sandbox
            .walk_dir(&base_path)
            .into_iter()
            .filter_map(|e| e.ok())
        {
            if results.len() >= MAX_MATCH_RESULTS || files_scanned >= MAX_FILES_SCANNED {
                break;
            }

            let path = entry.path();
            if !path.is_file() {
                continue;
            }

            files_scanned += 1;

            let validated_path = self
                .sandbox
                .validate_walk_entry(path)
                .map_err(io_sandbox_error)?;

            let relative_path = validated_path
                .strip_prefix(self.sandbox.root())
                .unwrap_or(&validated_path);

            if file_pattern.matches_path(relative_path)
                && let Ok(content) = fs::read_to_string(&validated_path)
            {
                for (line_num, line) in content.lines().enumerate() {
                    if regex.is_match(line) {
                        results.push(format!(
                            "{}:{}: {}",
                            relative_path.display(),
                            line_num + 1,
                            line.trim()
                        ));
                        if results.len() >= MAX_MATCH_RESULTS {
                            break;
                        }
                    }
                }
            }
        }

        if results.is_empty() {
            Ok("No matches found.".to_string().into())
        } else {
            Ok(format!(
                "Found {} matches (showing up to {}, scanned up to {} files):\n{}",
                results.len(),
                MAX_MATCH_RESULTS,
                MAX_FILES_SCANNED,
                results.join("\n")
            )
            .into())
        }
    }
}

#[derive(Serialize, Deserialize, ToolInput, Debug)]
pub struct AnalyzeCodeArgs {
    #[input(
        description = "Relative path to the file or directory to analyze within the workspace"
    )]
    path: String,
    #[input(description = "Type of analysis: 'structure', 'complexity', 'dependencies'")]
    analysis_type: String,
}

#[tool(
    name = "AnalyzeCodeTool",
    description = "Analyze code structure, complexity, or dependencies within the workspace sandbox",
    input = AnalyzeCodeArgs,
)]
pub struct AnalyzeCodeTool {
    sandbox: FilesystemSandbox,
}

impl AnalyzeCodeTool {
    pub fn with_sandbox(sandbox: FilesystemSandbox) -> Self {
        Self { sandbox }
    }
}

#[async_trait]
impl ToolRuntime for AnalyzeCodeTool {
    async fn execute(&self, args: serde_json::Value) -> Result<serde_json::Value, ToolCallError> {
        let args: AnalyzeCodeArgs = serde_json::from_value(args)?;

        let path = self
            .sandbox
            .resolve_relative(&args.path)
            .map_err(io_sandbox_error)?;

        if !path.exists() {
            return Err(ToolCallError::RuntimeError(
                format!("Path {} does not exist", path.display()).into(),
            ));
        }

        let path = self
            .sandbox
            .ensure_resolved(&path)
            .map_err(io_sandbox_error)?;

        match args.analysis_type.as_str() {
            "structure" => Ok(analyze_structure(&self.sandbox, &path)?.into()),
            "complexity" => Ok(analyze_complexity(&self.sandbox, &path)?.into()),
            "dependencies" => Ok(analyze_dependencies(&self.sandbox, &path)?.into()),
            _ => Err(ToolCallError::RuntimeError(
                "Invalid analysis type. Choose 'structure', 'complexity', or 'dependencies'".into(),
            )),
        }
    }
}

fn analyze_structure(sandbox: &FilesystemSandbox, path: &Path) -> Result<String, ToolCallError> {
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
        for entry in sandbox.walk_dir(path).into_iter().filter_map(|e| e.ok()) {
            let entry_path = entry.path();
            let validated_path = sandbox
                .validate_walk_entry(entry_path)
                .map_err(io_sandbox_error)?;

            if validated_path.is_file() {
                file_count += 1;
                if let Ok(content) = fs::read_to_string(&validated_path) {
                    total_lines += content.lines().count();
                }
                if let Some(ext) = validated_path.extension() {
                    *extensions
                        .entry(ext.to_string_lossy().to_string())
                        .or_insert(0) += 1;
                }
            } else if validated_path.is_dir() && validated_path != path {
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

fn collect_source_files(
    sandbox: &FilesystemSandbox,
    path: &Path,
) -> Result<Vec<std::path::PathBuf>, ToolCallError> {
    let mut files = Vec::new();

    if path.is_file() {
        files.push(path.to_path_buf());
        return Ok(files);
    }

    for entry in sandbox.walk_dir(path).into_iter().filter_map(|e| e.ok()) {
        let entry_path = entry.path();
        let validated_path = sandbox
            .validate_walk_entry(entry_path)
            .map_err(io_sandbox_error)?;

        if validated_path.is_file() {
            files.push(validated_path);
        }
    }

    Ok(files)
}

fn relative_display(sandbox: &FilesystemSandbox, path: &Path) -> String {
    path.strip_prefix(sandbox.root())
        .unwrap_or(path)
        .display()
        .to_string()
}

fn count_regex_matches(content: &str, pattern: &str) -> usize {
    Regex::new(pattern)
        .map(|regex| regex.find_iter(content).count())
        .unwrap_or(0)
}

fn analyze_complexity(sandbox: &FilesystemSandbox, path: &Path) -> Result<String, ToolCallError> {
    let files = collect_source_files(sandbox, path)?;
    if files.is_empty() {
        return Ok("Complexity Analysis:\nNo source files found.".to_string());
    }

    let file_count = files.len();
    let mut report = String::from("Complexity Analysis:\n");
    let mut total_lines = 0usize;
    let mut total_functions = 0usize;
    let mut total_complexity = 0usize;

    for file in files {
        let content = fs::read_to_string(&file).map_err(|e| {
            ToolCallError::RuntimeError(
                format!("Failed to read {}: {}", relative_display(sandbox, &file), e).into(),
            )
        })?;

        let lines = content.lines().count();
        let functions = count_regex_matches(
            &content,
            r"(?m)^\s*(pub\s+)?(async\s+)?fn\s+\w+|^\s*def\s+\w+|^\s*function\s+\w+",
        );
        let decision_points = count_regex_matches(
            &content,
            r"\b(if|else if|for|while|match|case|catch|&&|\|\|)\b|\?",
        );
        let complexity = 1 + decision_points;

        total_lines += lines;
        total_functions += functions;
        total_complexity += complexity;

        report.push_str(&format!(
            "\n{}:\n  Lines: {}\n  Functions: {}\n  Estimated cyclomatic complexity: {}\n",
            relative_display(sandbox, &file),
            lines,
            functions,
            complexity
        ));
    }

    report.push_str(&format!(
        "\nTotals:\n  Files: {}\n  Lines: {}\n  Functions: {}\n  Combined estimated cyclomatic complexity: {}\n",
        file_count,
        total_lines,
        total_functions,
        total_complexity
    ));

    Ok(report)
}

fn extract_dependencies(content: &str) -> Vec<String> {
    let patterns = [
        r"(?m)^\s*use\s+([^;{]+)",
        r"(?m)^\s*mod\s+(\w+)",
        r"(?m)^\s*import\s+(.+)",
        r"(?m)^\s*from\s+([^\s]+)\s+import",
        r#"(?m)require\(\s*['"]([^'"]+)['"]\s*\)"#,
    ];

    let mut deps = Vec::new();
    for pattern in patterns {
        if let Ok(regex) = Regex::new(pattern) {
            for cap in regex.captures_iter(content) {
                if let Some(dep) = cap.get(1) {
                    let value = dep.as_str().trim();
                    if !value.is_empty() {
                        deps.push(value.to_string());
                    }
                }
            }
        }
    }
    deps.sort();
    deps.dedup();
    deps
}

fn analyze_dependencies(sandbox: &FilesystemSandbox, path: &Path) -> Result<String, ToolCallError> {
    let files = collect_source_files(sandbox, path)?;
    if files.is_empty() {
        return Ok("Dependency Analysis:\nNo source files found.".to_string());
    }

    let mut report = String::from("Dependency Analysis:\n");
    let mut all_dependencies = Vec::new();

    for file in files {
        let content = fs::read_to_string(&file).map_err(|e| {
            ToolCallError::RuntimeError(
                format!("Failed to read {}: {}", relative_display(sandbox, &file), e).into(),
            )
        })?;

        let deps = extract_dependencies(&content);
        if deps.is_empty() {
            continue;
        }

        report.push_str(&format!("\n{}:\n", relative_display(sandbox, &file)));
        for dep in &deps {
            report.push_str(&format!("  - {}\n", dep));
            all_dependencies.push(dep.clone());
        }
    }

    all_dependencies.sort();
    all_dependencies.dedup();

    if all_dependencies.is_empty() {
        report.push_str("\nNo import/use statements found.");
    } else {
        report.push_str(&format!(
            "\nUnique dependencies ({}):\n",
            all_dependencies.len()
        ));
        for dep in all_dependencies {
            report.push_str(&format!("  - {}\n", dep));
        }
    }

    Ok(report)
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

        let result = GrepTool::with_sandbox(FilesystemSandbox::new(&dir).expect("sandbox"))
            .execute(json!({
                "pattern": "needle",
                "file_pattern": "src/*.rs",
                "subdirectory": ".",
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

        let tool = GrepTool::with_sandbox(FilesystemSandbox::new(&dir).expect("sandbox"));

        let no_match = tool
            .execute(json!({
                "pattern": "missing",
                "file_pattern": "src/*.rs",
                "subdirectory": ".",
            }))
            .await
            .expect("grep should succeed even when there are no matches");
        assert_eq!(no_match, json!("No matches found."));

        let invalid_regex = tool
            .execute(json!({
                "pattern": "(",
                "file_pattern": "src/*.rs",
                "subdirectory": ".",
            }))
            .await
            .expect_err("invalid regex should fail");
        assert!(invalid_regex.to_string().contains("Invalid regex"));

        let invalid_glob = tool
            .execute(json!({
                "pattern": "present",
                "file_pattern": "[*.rs",
                "subdirectory": ".",
            }))
            .await
            .expect_err("invalid glob should fail");
        assert!(invalid_glob.to_string().contains("Invalid file pattern"));

        let missing_dir = tool
            .execute(json!({
                "pattern": "present",
                "file_pattern": "*.rs",
                "subdirectory": "missing",
            }))
            .await
            .expect_err("missing directory should fail");
        assert!(missing_dir.to_string().contains("does not exist"));

        let _ = fs::remove_dir_all(dir);
    }

    #[tokio::test]
    async fn grep_tool_rejects_path_outside_workspace() {
        let dir = temp_fixture_dir("grep-escape");
        let tool = GrepTool::with_sandbox(FilesystemSandbox::new(&dir).expect("sandbox"));

        let err = tool
            .execute(json!({
                "pattern": "needle",
                "file_pattern": "*.rs",
                "subdirectory": "../outside",
            }))
            .await
            .expect_err("traversal should fail");
        assert!(
            err.to_string().contains("traversal")
                || err.to_string().contains("not allowed")
                || err.to_string().contains("RuntimeError")
        );

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

        let tool = AnalyzeCodeTool::with_sandbox(FilesystemSandbox::new(&dir).expect("sandbox"));

        let structure = tool
            .execute(json!({
                "path": ".",
                "analysis_type": "structure",
            }))
            .await
            .expect("structure analysis should succeed");
        let summary = structure.as_str().expect("summary should be a string");
        assert!(summary.contains("Code Structure Analysis"));
        assert!(summary.contains("- Files: 2"));
        assert!(summary.contains(".rs: 1 files"));
        assert!(summary.contains(".py: 1 files"));

        let sandbox = FilesystemSandbox::new(&dir).expect("sandbox");
        let single_file_summary = analyze_structure(&sandbox, &dir.join("src/lib.rs"))
            .expect("single-file structure analysis should succeed");
        assert!(single_file_summary.contains("- Files: 1"));
        assert!(single_file_summary.contains(".rs: 1 files"));

        let complexity = tool
            .execute(json!({
                "path": ".",
                "analysis_type": "complexity",
            }))
            .await
            .expect("complexity analysis should succeed");
        assert!(
            complexity
                .as_str()
                .expect("complexity output should be a string")
                .contains("Estimated cyclomatic complexity")
        );
        assert!(complexity.as_str().unwrap().contains("src/lib.rs"));

        let dependencies = tool
            .execute(json!({
                "path": ".",
                "analysis_type": "dependencies",
            }))
            .await
            .expect("dependency analysis should succeed");
        assert!(
            dependencies
                .as_str()
                .expect("dependency output should be a string")
                .contains("Dependency Analysis")
        );

        let invalid_type = tool
            .execute(json!({
                "path": ".",
                "analysis_type": "unknown",
            }))
            .await
            .expect_err("unknown analysis type should fail");
        assert!(invalid_type.to_string().contains("Invalid analysis type"));

        let missing_path = tool
            .execute(json!({
                "path": "missing",
                "analysis_type": "structure",
            }))
            .await
            .expect_err("missing path should fail");
        assert!(missing_path.to_string().contains("does not exist"));

        let _ = fs::remove_dir_all(dir);
    }
}
