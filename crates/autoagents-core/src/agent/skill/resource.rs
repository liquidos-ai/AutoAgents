use cap_fs_ext::{DirExt, FollowSymlinks, OpenOptionsFollowExt};
use cap_std::fs::{Dir, File, OpenOptions};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

#[derive(Debug)]
pub(crate) struct SkillResourceBoundary {
    directory: Dir,
    path: PathBuf,
}

impl SkillResourceBoundary {
    pub(crate) fn open(path: PathBuf) -> io::Result<Self> {
        let directory = Dir::open_ambient_dir(&path, cap_std::ambient_authority())?;
        Ok(Self { directory, path })
    }

    pub(crate) fn open_skill(&self, skill_directory: &Path) -> io::Result<SkillResourceDirectory> {
        let relative = skill_directory.strip_prefix(&self.path).map_err(|_| {
            io::Error::new(
                io::ErrorKind::PermissionDenied,
                "skill directory is outside its resource boundary",
            )
        })?;
        if relative.as_os_str().is_empty() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidInput,
                "skill directory must be below its resource boundary",
            ));
        }
        self.directory
            .open_dir_nofollow(relative)
            .map(SkillResourceDirectory::new)
    }
}

#[derive(Clone)]
pub(crate) struct SkillResourceDirectory {
    directory: Arc<Dir>,
}

impl SkillResourceDirectory {
    fn new(directory: Dir) -> Self {
        Self {
            directory: Arc::new(directory),
        }
    }

    pub(crate) fn open_file(&self, relative: &Path) -> io::Result<File> {
        let mut options = OpenOptions::new();
        options.read(true).follow(FollowSymlinks::No);
        self.directory.open_with(relative, &options)
    }
}

impl std::fmt::Debug for SkillResourceDirectory {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SkillResourceDirectory")
            .finish_non_exhaustive()
    }
}

impl PartialEq for SkillResourceDirectory {
    fn eq(&self, _other: &Self) -> bool {
        true
    }
}

impl Eq for SkillResourceDirectory {}

#[cfg(all(test, unix))]
mod tests {
    use super::SkillResourceBoundary;
    use std::fs;
    use std::io::ErrorKind;
    use std::os::unix::fs::symlink;

    #[test]
    fn captured_directory_is_not_redirected_by_a_path_replacement() {
        let fixture = tempfile::tempdir().expect("fixture");
        let boundary_path = fixture.path().join("boundary");
        let original = boundary_path.join("skill");
        fs::create_dir_all(&original).expect("original skill directory");
        fs::write(original.join("resource.md"), "trusted content").expect("original resource");
        let boundary = SkillResourceBoundary::open(boundary_path.clone()).expect("boundary");
        let captured = boundary.open_skill(&original).expect("captured directory");

        let displaced = boundary_path.join("displaced");
        fs::rename(&original, &displaced).expect("displace original directory");
        let outside = fixture.path().join("outside");
        fs::create_dir(&outside).expect("outside directory");
        fs::write(outside.join("resource.md"), "untrusted content").expect("outside resource");
        symlink(&outside, &original).expect("replacement symlink");

        let content = std::io::read_to_string(
            captured
                .open_file(std::path::Path::new("resource.md"))
                .expect("captured resource"),
        )
        .expect("resource content");

        assert_eq!(content, "trusted content");
    }

    #[test]
    fn resource_boundary_rejects_itself_and_outside_directories() {
        let fixture = tempfile::tempdir().expect("fixture");
        let boundary_path = fixture.path().join("boundary");
        let first_skill = boundary_path.join("first");
        let second_skill = boundary_path.join("second");
        fs::create_dir_all(&first_skill).expect("first skill directory");
        fs::create_dir_all(&second_skill).expect("second skill directory");
        let boundary = SkillResourceBoundary::open(boundary_path.clone()).expect("boundary");

        let outside_error = boundary
            .open_skill(fixture.path())
            .expect_err("outside directory must be rejected");
        assert_eq!(outside_error.kind(), ErrorKind::PermissionDenied);

        let boundary_error = boundary
            .open_skill(&boundary_path)
            .expect_err("boundary root is not a skill directory");
        assert_eq!(boundary_error.kind(), ErrorKind::InvalidInput);

        let first = boundary.open_skill(&first_skill).expect("first skill");
        let second = boundary.open_skill(&second_skill).expect("second skill");
        assert_eq!(first, second);
        assert!(format!("{first:?}").contains("SkillResourceDirectory"));
    }
}
