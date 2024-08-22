import jax
import jax.numpy as jnp
import pytest

from liesel_ptm.model import OnionPTMLocScale
from liesel_ptm.nodes import LinearTerm, VarInverseGamma

key = jax.random.PRNGKey(42)
k1, k2 = jax.random.split(key)

beta = jnp.array([0.0, 1.0, -0.5])
X = jax.random.uniform(k1, shape=(100, 3))
y = X @ beta + jax.random.normal(k2, shape=(X.shape[0],))


class TestOnionPTMLocScale:
    def test_init(self) -> None:
        model = OnionPTMLocScale(
            y=y,
            nparam=15,
            tau2=VarInverseGamma(0.2, concentration=2.0, scale=0.5, name="tau2"),
        )

        assert model is not None

    def test_optimize_start_values(self) -> None:
        model = OnionPTMLocScale(
            y=y,
            nparam=15,
            tau2=VarInverseGamma(0.2, concentration=2.0, scale=0.5, name="tau2"),
        )

        model.loc_model += LinearTerm(x=X, name="lin")

        res1 = model.optimize_locscale(atol=0.001)
        _ = model.optimize_transformation()

        Xd = jnp.c_[jnp.ones(100), X]
        beta_hat = jnp.linalg.inv(Xd.T @ Xd) @ Xd.T @ y

        assert model.loc_intercept.value == pytest.approx(beta_hat[0], abs=0.05)
        assert jnp.allclose(res1.position["lin_coef"], beta_hat[1:], atol=0.05)

    def test_setup_engine_builder(self) -> None:
        model = OnionPTMLocScale(
            y=y,
            nparam=15,
            tau2=VarInverseGamma(0.2, concentration=2.0, scale=0.5, name="tau2"),
        )

        model.loc_model += LinearTerm(x=X, name="lin")

        eb = model.setup_engine_builder(seed=1, num_chains=4)
        assert eb is not None

    @pytest.mark.mcmc
    def test_mcmc(self) -> None:
        model = OnionPTMLocScale(
            y=y,
            nparam=15,
            tau2=VarInverseGamma(0.2, concentration=2.0, scale=0.5, name="tau2"),
        )

        model.loc_model += LinearTerm(x=X, name="lin")

        model.optimize_locscale()
        model.optimize_transformation()

        eb = model.setup_engine_builder(seed=1, num_chains=4)
        eb.set_duration(warmup_duration=500, posterior_duration=100)
        engine = eb.build()
        engine.sample_all_epochs()

        results = engine.get_results()
        samples = results.get_posterior_samples()

        for values in samples.values():
            assert not jnp.any(jnp.isnan(values))

        posterior_means_lin_coef = samples["lin_coef"].mean(axis=(0, 1))

        Xd = jnp.c_[jnp.ones(100), X]
        beta_hat = jnp.linalg.inv(Xd.T @ Xd) @ Xd.T @ y

        assert jnp.allclose(posterior_means_lin_coef, beta_hat[1:], atol=0.2)

        intercept = samples["loc_intercept"].mean(axis=(0, 1))
        assert intercept == pytest.approx(beta_hat[0], abs=0.1)

        scale = samples["scale_intercept_exp"].mean(axis=(0, 1))
        assert scale == pytest.approx(1.0, abs=0.1)

        log_increments = model.coef.log_increments.predict(samples).mean(axis=(0, 1))
        assert jnp.allclose(
            jnp.exp(log_increments), jnp.exp(log_increments).mean(), atol=0.2
        )

    def test_predict_loc(self) -> None:
        model = OnionPTMLocScale(
            y=y,
            nparam=15,
            tau2=VarInverseGamma(0.2, concentration=2.0, scale=0.5, name="tau2"),
        )

        model.loc_model += LinearTerm(x=X, name="lin")

        samples = {"lin_coef": jax.random.normal(key, shape=(4, 20, 3))}
        loc = model.predict_loc(samples, lin=X)
        assert not jnp.any(jnp.isnan(loc))
        assert loc.shape == (4, 20, model.response.value.shape[0])

    def test_predict_scale(self) -> None:
        model = OnionPTMLocScale(
            y=y,
            nparam=15,
            tau2=VarInverseGamma(0.2, concentration=2.0, scale=0.5, name="tau2"),
        )

        model.log_scale_model += LinearTerm(x=X, name="lin")

        samples = {"lin_coef": jax.random.normal(key, shape=(4, 20, 3))}
        scale = model.predict_scale(samples, lin=X)
        assert not jnp.any(jnp.isnan(scale))
        assert scale.shape == (4, 20, model.response.value.shape[0])

    def test_init_dist(self) -> None:
        model = OnionPTMLocScale(
            y=y,
            nparam=15,
            tau2=VarInverseGamma(0.2, concentration=2.0, scale=0.5, name="tau2"),
        )

        model.loc_model += LinearTerm(x=X, name="lin")

        model.init_dist()

        samples = {"lin_coef": jax.random.normal(key, shape=(4, 20, 3))}
        samples["shape_coef_latent_param"] = jax.random.normal(key, shape=(4, 20, 15))
        samples["tau2_transformed"] = jax.random.normal(key, (4, 20))

        model.init_dist(samples, lin=X)

        assert True

    def test_augmented_samples(self) -> None:
        model = OnionPTMLocScale(
            y=y,
            nparam=15,
            tau2=VarInverseGamma(0.2, concentration=2.0, scale=0.5, name="tau2"),
        )

        model.loc_model += LinearTerm(x=X, name="lin")

        model.init_dist()

        samples = {"lin_coef": jax.random.normal(key, shape=(4, 20, 3))}

        dist = model.init_dist(samples, lin=X)
        assert dist.coef.shape == model.coef.value.shape

        n = model.response.value.shape[0]
        assert dist.apriori_distribution_kwargs["loc"].shape == (4, 20, n)

        assert dist.log_prob(model.response.value).shape == (4, 20, n)

    def test_augmented_kwargs(self) -> None:
        model = OnionPTMLocScale(
            y=y,
            nparam=15,
            tau2=VarInverseGamma(0.2, concentration=2.0, scale=0.5, name="tau2"),
        )

        model.loc_model += LinearTerm(x=X, name="lin")

        model.init_dist()

        samples = {"lin_coef": jax.random.normal(key, shape=(4, 20, 3))}

        dist = model.init_dist(samples)
        lp1 = dist.log_prob(model.response.value)

        dist = model.init_dist(samples, lin=X)
        lp2 = dist.log_prob(model.response.value)

        assert jnp.allclose(lp1, lp2)

    def test_waic(self) -> None:
        model = OnionPTMLocScale(
            y=y,
            nparam=15,
            tau2=VarInverseGamma(0.2, concentration=2.0, scale=0.5, name="tau2"),
        )

        model.loc_model += LinearTerm(x=X, name="lin")
        samples = {"lin_coef": jax.random.normal(key, shape=(4, 20, 3))}

        waic = model.waic(samples)
        assert waic.isna().sum().sum() == 0

    def test_summarise_density_by_quantiles(self) -> None:
        model = OnionPTMLocScale(
            y=y,
            nparam=15,
            tau2=VarInverseGamma(0.2, concentration=2.0, scale=0.5, name="tau2"),
        )

        model.loc_model += LinearTerm(x=X, name="lin")
        samples = {"lin_coef": jax.random.normal(key, shape=(4, 20, 3))}

        df = model.summarise_density_by_quantiles(
            y=jnp.linspace(-3, 3, 101), samples=samples, lin=jnp.zeros((1, 3))
        )

        assert df.shape == (101, 19)

    def test_summarise_density_by_samples(self) -> None:
        model = OnionPTMLocScale(
            y=y,
            nparam=15,
            tau2=VarInverseGamma(0.2, concentration=2.0, scale=0.5, name="tau2"),
        )

        model.loc_model += LinearTerm(x=X, name="lin")
        samples = {"lin_coef": jax.random.normal(key, shape=(4, 20, 3))}

        df = model.summarise_density_by_samples(
            key=1,
            y=jnp.linspace(-3, 3, 101),
            samples=samples,
            lin=jnp.zeros((101, 3)),
            n=100,
        )

        assert df.shape == (101 * 100, 11)

    def test_summarise_transformation_by_quantiles(self) -> None:
        model = OnionPTMLocScale(
            y=y,
            nparam=15,
            tau2=VarInverseGamma(0.2, concentration=2.0, scale=0.5, name="tau2"),
        )

        model.loc_model += LinearTerm(x=X, name="lin")
        samples = {"lin_coef": jax.random.normal(key, shape=(4, 20, 3))}
        samples["shape_coef_latent_param"] = jax.random.normal(key, shape=(4, 20, 15))

        df = model.summarise_transformation_by_quantiles(
            residuals=jnp.linspace(-3, 3, 101),
            samples=samples,
        )

        assert df.shape == (101, 16)
