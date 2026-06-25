use proc_macro2::Span;
use quote::format_ident;
use syn::{Error, Path, Result, parse_quote};

enum DependencyLayout {
    DirectCore { core: Path },
    Facade { core: Path, async_trait: Path },
}

/// Resolves core and `async_trait` paths for `#[derive(AgentHooks)]` with a single layout lookup.
pub(crate) fn resolve_agent_hooks_paths() -> Result<(Path, Path)> {
    match resolve_layout()? {
        DependencyLayout::DirectCore { core } => {
            let async_trait = resolve_direct_async_trait_path()?;
            Ok((core, async_trait))
        }
        DependencyLayout::Facade { core, async_trait } => Ok((core, async_trait)),
    }
}

/// Resolves the core crate path for generated code in the consumer crate.
pub(crate) fn resolve_core_path() -> Result<Path> {
    Ok(match resolve_layout()? {
        DependencyLayout::DirectCore { core } | DependencyLayout::Facade { core, .. } => core,
    })
}

fn resolve_layout() -> Result<DependencyLayout> {
    if let Ok(core_name) = proc_macro_crate::crate_name("autoagents-core") {
        return Ok(DependencyLayout::DirectCore {
            core: crate_name_to_path(core_name, "autoagents-core"),
        });
    }

    if let Ok(facade_name) = proc_macro_crate::crate_name("autoagents") {
        let facade = crate_name_to_path(facade_name, "autoagents");
        let core = path_with_suffix(&facade, "core");
        let async_trait = path_with_suffix(&facade, "async_trait");
        return Ok(DependencyLayout::Facade { core, async_trait });
    }

    Err(Error::new(
        Span::call_site(),
        "autoagents-derive requires either `autoagents` or `autoagents-core` as a direct dependency",
    ))
}

fn resolve_direct_async_trait_path() -> Result<Path> {
    let name = proc_macro_crate::crate_name("async-trait").map_err(|_| {
        Error::new(
            Span::call_site(),
            "when using `autoagents-core` directly, `async-trait` must also be a direct dependency for `#[derive(AgentHooks)]`",
        )
    })?;
    let crate_path = crate_name_to_path(name, "async-trait");
    Ok(path_with_suffix(&crate_path, "async_trait"))
}

fn crate_name_to_path(name: proc_macro_crate::FoundCrate, itself_fallback: &str) -> Path {
    let ident_str = match name {
        proc_macro_crate::FoundCrate::Itself => itself_fallback.replace('-', "_"),
        proc_macro_crate::FoundCrate::Name(n) => n.replace('-', "_"),
    };
    let ident = format_ident!("{}", ident_str);
    parse_quote!(#ident)
}

fn path_with_suffix(base: &Path, suffix: &str) -> Path {
    let base_ident = base
        .segments
        .last()
        .expect("resolved crate path must not be empty")
        .ident
        .clone();
    let suffix = format_ident!("{}", suffix);
    parse_quote!(#base_ident::#suffix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_name_to_path_replaces_hyphens() {
        let path = crate_name_to_path(
            proc_macro_crate::FoundCrate::Name("autoagents-core".to_string()),
            "autoagents-core",
        );
        assert_eq!(path.segments.len(), 1);
        assert_eq!(path.segments[0].ident, "autoagents_core");
    }

    #[test]
    fn crate_name_to_path_itself_uses_fallback() {
        let path = crate_name_to_path(proc_macro_crate::FoundCrate::Itself, "autoagents-core");
        assert_eq!(path.segments[0].ident, "autoagents_core");
    }

    #[test]
    fn path_with_suffix_appends_segment() {
        let facade = crate_name_to_path(
            proc_macro_crate::FoundCrate::Name("autoagents".to_string()),
            "autoagents",
        );
        let core = path_with_suffix(&facade, "core");
        assert_eq!(core.segments.len(), 2);
        assert_eq!(core.segments[0].ident, "autoagents");
        assert_eq!(core.segments[1].ident, "core");
    }
}
