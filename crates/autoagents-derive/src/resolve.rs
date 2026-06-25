use proc_macro2::Span;
use quote::format_ident;
use syn::{Error, Path, Result};

/// Resolved paths for generated code in the consumer crate.
pub(crate) struct ResolvedPaths {
    /// Path to core types, e.g. `autoagents_core` or `autoagents::core`.
    pub core: Path,
    /// Path to the `async_trait` attribute macro.
    pub async_trait: Path,
}

/// Resolves crate paths from the calling crate's direct dependencies.
///
/// Prefers `autoagents-core` when present, otherwise falls back to the `autoagents` facade.
pub(crate) fn resolve_paths() -> Result<ResolvedPaths> {
    if let Ok(core_name) = proc_macro_crate::crate_name("autoagents-core") {
        let core = crate_name_to_path(core_name);
        let async_trait = resolve_async_trait_path()?;
        return Ok(ResolvedPaths { core, async_trait });
    }

    if let Ok(facade_name) = proc_macro_crate::crate_name("autoagents") {
        let facade = crate_name_to_path(facade_name);
        let core = {
            let mut path = facade.clone();
            path.segments.push(format_ident!("core").into());
            path
        };
        let async_trait = {
            let mut path = facade;
            path.segments.push(format_ident!("async_trait").into());
            path
        };
        return Ok(ResolvedPaths { core, async_trait });
    }

    Err(Error::new(
        Span::call_site(),
        "autoagents-derive requires either `autoagents` or `autoagents-core` as a direct dependency",
    ))
}

fn resolve_async_trait_path() -> Result<Path> {
    let name = proc_macro_crate::crate_name("async-trait").map_err(|_| {
        Error::new(
            Span::call_site(),
            "when using `autoagents-core` directly, `async-trait` must also be a direct dependency",
        )
    })?;
    let mut path = crate_name_to_path(name);
    path.segments.push(format_ident!("async_trait").into());
    Ok(path)
}

fn crate_name_to_path(name: proc_macro_crate::FoundCrate) -> Path {
    let ident = match name {
        proc_macro_crate::FoundCrate::Itself => format_ident!("autoagents_derive"),
        proc_macro_crate::FoundCrate::Name(n) => format_ident!("{}", n.replace('-', "_")),
    };
    Path::from(ident)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crate_name_to_path_replaces_hyphens() {
        let path = crate_name_to_path(proc_macro_crate::FoundCrate::Name(
            "autoagents-core".to_string(),
        ));
        assert_eq!(path.segments.len(), 1);
        assert_eq!(path.segments[0].ident, "autoagents_core");
    }
}
