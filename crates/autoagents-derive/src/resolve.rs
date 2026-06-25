use proc_macro2::Span;
use quote::format_ident;
use syn::{Error, Path, PathArguments, PathSegment, Result};

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
        let core = {
            let mut path = facade.clone();
            path.segments
                .push(path_segment_from_ident(format_ident!("core")));
            path
        };
        let async_trait = {
            let mut path = facade;
            path.segments
                .push(path_segment_from_ident(format_ident!("async_trait")));
            path
        };
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
    let mut path = crate_name_to_path(name, "async-trait");
    path.segments
        .push(path_segment_from_ident(format_ident!("async_trait")));
    Ok(path)
}

fn path_segment_from_ident(ident: proc_macro2::Ident) -> PathSegment {
    PathSegment {
        ident,
        arguments: PathArguments::None,
    }
}

fn crate_name_to_path(name: proc_macro_crate::FoundCrate, itself_fallback: &str) -> Path {
    let ident = match name {
        proc_macro_crate::FoundCrate::Itself => {
            format_ident!("{}", itself_fallback.replace('-', "_"))
        }
        proc_macro_crate::FoundCrate::Name(n) => format_ident!("{}", n.replace('-', "_")),
    };
    Path {
        leading_colon: None,
        segments: {
            let mut segments = syn::punctuated::Punctuated::new();
            segments.push(path_segment_from_ident(ident));
            segments
        },
    }
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
}
