//! The macro that defines the tests of one sequence.

/// Define a sequence's tests: on disk, on the server under `minio_tests`, on AWS under `s3_tests`.
macro_rules! parity_sequence {
    ($sequence:ident) => {
        mod $sequence {
            use crate::System as _;

            #[tokio::test(flavor = "current_thread")]
            async fn on_disk() {
                let real = super::RealSystem::new();
                let tmp = real.create_temp_dir().await.unwrap();
                super::$sequence(&real, tmp.path()).await;
            }

            #[cfg(minio_tests)]
            #[tokio::test(flavor = "current_thread")]
            async fn on_the_server() {
                super::run_on_the_server(stringify!($sequence), super::$sequence).await;
            }

            #[cfg(s3_tests)]
            #[tokio::test(flavor = "current_thread")]
            async fn on_aws() {
                super::run_on_aws(stringify!($sequence), super::$sequence).await;
            }
        }
    };
}
