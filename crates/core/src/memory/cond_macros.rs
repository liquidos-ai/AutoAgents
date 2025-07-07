#[macro_export]
macro_rules! cond {
    (any)                       => { $crate::memory::MessageCondition::Any };
    (eq          $v:literal)    => { $crate::memory::MessageCondition::Eq($v.into()) };
    (contains    $v:literal)    => { $crate::memory::MessageCondition::Contains($v.into()) };
    (not_contains $v:literal)   => { $crate::memory::MessageCondition::NotContains($v.into()) };
    (role        $v:literal)    => { $crate::memory::MessageCondition::RoleIs($v.into()) };
    (role_not    $v:literal)    => { $crate::memory::MessageCondition::RoleNot($v.into()) };
    (len_gt      $v:literal)    => { $crate::memory::MessageCondition::LenGt($v) };
    (regex       $v:literal)    => { $crate::memory::MessageCondition::Regex($v.into()) };

    ($left:tt && $($rest:tt)+) => {
        $crate::memory::MessageCondition::All(vec![
            $crate::cond!($left),
            $crate::cond!($($rest)+),
        ])
    };
    ($left:tt || $($rest:tt)+) => {
        $crate::memory::MessageCondition::AnyOf(vec![
            $crate::cond!($left),
            $crate::cond!($($rest)+),
        ])
    };

    ( ( $($inner:tt)+ ) ) => { $crate::cond!($($inner)+) };
}

#[macro_export]
macro_rules! on {
    ($builder:expr , $role:literal => $($cond:tt)+) => {
        $builder.on_message_from_with_trigger($role, $crate::cond!($($cond)+))
    };
}
