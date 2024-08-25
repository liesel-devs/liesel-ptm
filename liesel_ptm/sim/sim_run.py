import logging
import logging.handlers
import os
from pathlib import Path

import click
import jax.numpy as jnp
import tensorflow_probability.substrates.jax.distributions as tfd
from liesel.logging import add_file_handler
from liesel.model import Dist

import liesel_ptm as ptm
from liesel_ptm.sim.sim_bctm import one_run as run_bctm
from liesel_ptm.sim.sim_ptm import one_run as run_ptm
from liesel_ptm.sim.sim_ptm_onion import one_run as run_ptm_onion


def emit(self, record):
    """
    Overwrite the logging.handlers.SMTPHandler.emit function with SMTP_SSL.
    Emit a record.
    Format the record and send it to the specified addressees.
    """
    try:
        import smtplib
        from email.utils import formatdate

        port = self.mailport
        if not port:
            port = smtplib.SMTP_PORT
        smtp = smtplib.SMTP_SSL(self.mailhost, port, timeout=self.timeout)
        msg = self.format(record)
        msg = "From: {}\r\nTo: {}\r\nSubject: {}\r\nDate: {}\r\n\r\n{}".format(
            self.fromaddr,
            ", ".join(self.toaddrs),
            self.getSubject(record),
            formatdate(),
            msg,
        )
        if self.username:
            smtp.ehlo()
            smtp.login(self.username, self.password)
        smtp.sendmail(self.fromaddr, self.toaddrs, msg)
        smtp.quit()
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception:
        self.handleError(record)


logging.handlers.SMTPHandler.emit = emit  # type: ignore

# Email configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 465
SMTP_USERNAME: str = "johannes.brachem@gmail.com"
SMTP_PASSWORD: str = os.environ.get("JB_GMAIL_PW", default="")
SENDER_EMAIL = "johannes.brachem@gmail.com"
RECIPIENT_EMAIL = "jo@jostats.de"

# Set up email handler
email_handler = logging.handlers.SMTPHandler(
    mailhost=(SMTP_SERVER, SMTP_PORT),
    fromaddr=SENDER_EMAIL,
    toaddrs=[RECIPIENT_EMAIL],
    subject="Log Message",
    credentials=(SMTP_USERNAME, SMTP_PASSWORD),
    secure=None,
)


mailog = logging.getLogger("mail")
mailog.setLevel(logging.INFO)
mailog.addHandler(email_handler)
logger = logging.getLogger("sim")

PATH = Path("/home/brachem/projects/2024-02-ptm_one_big_run")


def setup_logging(seed: int, path: Path | str, data: str):
    liesel_logger = logging.getLogger("liesel")
    liesel_stdout = liesel_logger.handlers[0]
    ptm_logger = logging.getLogger("liesel_ptm")

    try:
        simlogger_stdout = logger.handlers[0]
    except IndexError:
        simlogger_stdout = None

    try:
        ptmlogger_stdout = ptm_logger.handlers[0]
    except IndexError:
        ptmlogger_stdout = None

    path = Path(path).resolve()
    logs = path
    logs.mkdir(exist_ok=True, parents=False)
    logfile = logs / f"log-{data}-seed_{seed}.log"
    add_file_handler(path=logfile, level="info")
    add_file_handler(path=logfile, level="info", logger="sim")
    add_file_handler(path=logfile, level="info", logger="liesel_ptm")

    liesel_logger.removeHandler(liesel_stdout)
    if simlogger_stdout is not None:
        logger.removeHandler(simlogger_stdout)
    if ptmlogger_stdout is not None:
        ptm_logger.removeHandler(ptmlogger_stdout)


@click.command()
@click.option(
    "--seed", help="Seed for random number generation.", required=True, type=int
)
@click.option("--part", help="Which simulation part to run.", required=True, type=str)
@click.option("--data", help="Which dataset to use.", required=True, type=str)
@click.option("--n", help="Sample size to use.", required=True, type=int)
def run_simulation(seed: int, part: str, data: str, n: int):
    _run_simulation(seed=seed, part=part, data=data, n=n)


def _run_simulation(seed: int, part: str, data: str, n: int):
    logger.setLevel(logging.INFO)
    logdir = PATH / "logs" / f"{part}-N{n}"
    logdir.mkdir(exist_ok=True, parents=True)
    setup_logging(seed, logdir, data=data)

    DATA = [data]
    WARMUP = 1500
    POSTERIOR = 5000

    IGPRIOR = (
        '{"class": "VarInverseGamma", "value": 1.0, "concentration": 1.0, "scale":'
        " 0.01}"
    )
    SDPRIOR_05 = '{"class": "VarWeibull", "value": 1.0, "scale": 0.05}'
    SDPRIOR_01 = '{"class": "VarWeibull", "value": 0.5, "scale": 0.01}'

    mailog.info(f"STARTING: {part=}, {seed=}, {n=}.")

    scaling_prior = Dist(
        tfd.TruncatedNormal, loc=1.0, scale=0.1, low=0.01, high=jnp.inf
    )

    for data in DATA:
        logger.info(f"Starting iteration with {part=}, {n=}, {data=}, {seed=}.")
        mailog.info(f"Starting iteration with {part=}, {n=}, {data=}, {seed=}.")

        data_path = PATH / "_data" / data
        out_path = PATH
        cache_path = PATH / "cache"
        cache_path.mkdir(exist_ok=True)

        res = None

        # PTM SDPrior scale 0.05
        if part == "ptm-wb05-std":
            logger.warning("STARTING NEW RUN: PTMs with SD Prior, scale = 0.05")
            run_ptm(
                seed=seed,
                data_path=data_path,
                out_path=out_path,
                warmup=WARMUP,
                posterior=POSTERIOR,
                n=n,
                scale_terms=True,
                sample_transformation=True,
                identifier=f"ptm-wb05-std-{data}-N{n}",
                prior_tau2_covariates=IGPRIOR,
                prior_tau2_normalization=SDPRIOR_05,
                cache_path=cache_path,
                optimize_start_values=True,
                skip_if_results_exist=True,
                scale_after_transformation=True,
                id_data={"model": "ptm", "data": data},
            )

        if part == "bctm-01":
            logger.warning("STARTING NEW RUN: BCTM with 10,10 parameters")
            # BCTM 10,10 param
            run_bctm(
                seed=seed,
                data_path=data_path,
                out_path=out_path,
                warmup=WARMUP,
                posterior=POSTERIOR,
                identifier=f"bctm-10-x-{data}-N{n}",
                n=n,
                nparam_y=10,
                nparam_x=10,
                id_data={"model": "bctm", "data": data},
                cache_path=cache_path,
                skip_if_results_exist=True,
            )

        if part == "normal":
            logger.warning("STARTING NEW RUN: Normal model")
            # Normal model
            run_ptm(
                seed=seed,
                data_path=data_path,
                out_path=out_path,
                warmup=WARMUP,
                posterior=POSTERIOR,
                n=n,
                scale_terms=True,
                sample_transformation=False,
                identifier=f"normal-x-{data}-N{n}",
                prior_tau2_covariates=IGPRIOR,
                prior_tau2_normalization=SDPRIOR_01,
                cache_path=cache_path,
                optimize_start_values=True,
                skip_if_results_exist=True,
                id_data={"model": "normal", "data": data},
            )

        if part == "bctm-02":
            logger.warning("STARTING NEW RUN: BCTM with 15,15 parameters")
            # BCTM 15,15 param
            run_bctm(
                seed=seed,
                data_path=data_path,
                out_path=out_path,
                warmup=WARMUP,
                posterior=POSTERIOR,
                identifier=f"bctm-20-x-{data}-N{n}",
                n=n,
                nparam_y=15,
                nparam_x=15,
                id_data={"model": "bctm", "data": data},
                cache_path=cache_path,
                skip_if_results_exist=True,
            )

        if part == "ptm-wb05-bounded":
            logger.warning(
                "STARTING NEW RUN: PTMs with SD Prior, scale = 0.05, including a"
                " scaling factor."
            )
            # PTM SDPrior scale 0.01
            run_ptm(
                seed=seed,
                data_path=data_path,
                out_path=out_path,
                warmup=WARMUP,
                posterior=POSTERIOR,
                n=n,
                scale_terms=True,
                sample_transformation=True,
                identifier=f"ptm-wb05-bounded-{data}-N{n}",
                prior_tau2_covariates=IGPRIOR,
                prior_tau2_normalization=SDPRIOR_05,
                cache_path=cache_path,
                optimize_start_values=True,
                skip_if_results_exist=True,
                id_data={"model": "ptm", "data": data},
                scaling_factor=ptm.SymmetricallyBoundedScalar(0.3),
            )

        if part == "ptm-wb05-fixed":
            logger.warning(
                "STARTING NEW RUN: PTMs with SD Prior, scale = 0.05, with no scaling"
                " after transformation"
            )
            # PTM SDPrior scale 0.01
            run_ptm(
                seed=seed,
                data_path=data_path,
                out_path=out_path,
                warmup=WARMUP,
                posterior=POSTERIOR,
                n=n,
                scale_terms=True,
                sample_transformation=True,
                identifier=f"ptm-wb05-fixed-{data}-N{n}",
                prior_tau2_covariates=IGPRIOR,
                prior_tau2_normalization=SDPRIOR_05,
                cache_path=cache_path,
                optimize_start_values=True,
                skip_if_results_exist=True,
                id_data={"model": "ptm", "data": data},
                scale_after_transformation=False,
            )

        if part == "ptm-ridge-bounded":
            logger.warning(
                "STARTING NEW RUN: PTMs with SD Prior, scale = 0.05, including a"
                " scaling factor and using a ridge prior for the shape params."
            )
            # PTM SDPrior scale 0.01
            run_ptm(
                seed=seed,
                data_path=data_path,
                out_path=out_path,
                warmup=WARMUP,
                posterior=POSTERIOR,
                n=n,
                scale_terms=True,
                sample_transformation=True,
                identifier=f"ptm-ridge-bounded-{data}-N{n}",  # noqa
                prior_tau2_covariates=IGPRIOR,
                prior_tau2_normalization=SDPRIOR_05,
                cache_path=cache_path,
                optimize_start_values=True,
                skip_if_results_exist=True,
                id_data={"model": "ptm", "data": data},
                scaling_factor=ptm.SymmetricallyBoundedScalar(0.3),
                shape_param_prior=ptm.ShapePrior.RIDGE,
            )

        if part == "ptm-ridge-tnorm":
            logger.warning(
                "STARTING NEW RUN: PTMs with SD Prior, scale = 0.05, including a"
                " scaling factor and using a ridge prior for the shape params."
            )
            scaling_prior = Dist(
                tfd.TruncatedNormal, loc=1.0, scale=0.1, low=0.01, high=jnp.inf
            )
            # PTM SDPrior scale 0.01
            run_ptm(
                seed=seed,
                data_path=data_path,
                out_path=out_path,
                warmup=WARMUP,
                posterior=POSTERIOR,
                n=n,
                scale_terms=True,
                sample_transformation=True,
                identifier=f"ptm-ridge-tnorm-{data}-N{n}",  # noqa
                prior_tau2_covariates=IGPRIOR,
                prior_tau2_normalization=SDPRIOR_05,
                cache_path=cache_path,
                optimize_start_values=True,
                skip_if_results_exist=True,
                id_data={"model": "ptm", "data": data},
                scaling_factor=ptm.TransformedVar(
                    value=1.0, name="scaling_factor", prior=scaling_prior
                ),
                shape_param_prior=ptm.ShapePrior.RIDGE,
            )

        if part == "ptm-wb05-const":
            logger.warning(
                "STARTING NEW RUN: PTMs with SD Prior, scale = 0.05, including a"
                " scaling factor with constant prior."
            )
            # PTM SDPrior scale 0.01
            run_ptm(
                seed=seed,
                data_path=data_path,
                out_path=out_path,
                warmup=WARMUP,
                posterior=POSTERIOR,
                n=n,
                scale_terms=True,
                sample_transformation=True,
                identifier=f"ptm-wb05-const-{data}-N{n}",  # noqa
                prior_tau2_covariates=IGPRIOR,
                prior_tau2_normalization=SDPRIOR_05,
                cache_path=cache_path,
                optimize_start_values=True,
                skip_if_results_exist=True,
                id_data={"model": "ptm", "data": data},
                scaling_factor=ptm.ConstantPriorScalingFactor(),
            )

        if part == "ptm-wb05-tnorm":
            logger.warning(
                "STARTING NEW RUN: PTMs with SD Prior, scale = 0.05, including a"
                " scaling factor with truncated normal prior."
            )
            scaling_prior = Dist(
                tfd.TruncatedNormal, loc=1.0, scale=0.1, low=0.01, high=jnp.inf
            )
            # PTM SDPrior scale 0.01
            res = run_ptm(
                seed=seed,
                data_path=data_path,
                out_path=out_path,
                warmup=WARMUP,
                posterior=POSTERIOR,
                n=n,
                scale_terms=True,
                sample_transformation=True,
                identifier=f"ptm-wb05-tnorm-{data}-N{n}",  # noqa
                prior_tau2_covariates=IGPRIOR,
                prior_tau2_normalization=SDPRIOR_05,
                cache_path=cache_path,
                optimize_start_values=True,
                skip_if_results_exist=True,
                id_data={"model": "ptm", "data": data},
                scaling_factor=ptm.TransformedVar(
                    value=1.0, name="scaling_factor", prior=scaling_prior
                ),
            )

        if part == "ptm-ig-bounded":
            logger.warning("STARTING NEW RUN: PTMs with Inverse Gamma Prior")
            # PTM SDPrior scale 0.01
            run_ptm(
                seed=seed,
                data_path=data_path,
                out_path=out_path,
                warmup=WARMUP,
                posterior=POSTERIOR,
                n=n,
                scale_terms=True,
                sample_transformation=True,
                identifier=f"ptm-igprior-bounded-{data}-N{n}",
                prior_tau2_covariates=IGPRIOR,
                prior_tau2_normalization=IGPRIOR,
                cache_path=cache_path,
                optimize_start_values=True,
                skip_if_results_exist=True,
                id_data={"model": "ptm", "data": data},
                scaling_factor=ptm.SymmetricallyBoundedScalar(
                    0.3, name="scaling_factor"
                ),
            )

        if part == "ptm-ig-tnorm":
            logger.warning("STARTING NEW RUN: PTMs with Inverse Gamma Prior")
            # PTM SDPrior scale 0.01
            scaling_prior = Dist(
                tfd.TruncatedNormal, loc=1.0, scale=0.1, low=0.01, high=jnp.inf
            )
            run_ptm(
                seed=seed,
                data_path=data_path,
                out_path=out_path,
                warmup=WARMUP,
                posterior=POSTERIOR,
                n=n,
                scale_terms=True,
                sample_transformation=True,
                identifier=f"ptm-igprior-tnorm-{data}-N{n}",
                prior_tau2_covariates=IGPRIOR,
                prior_tau2_normalization=IGPRIOR,
                cache_path=cache_path,
                optimize_start_values=True,
                skip_if_results_exist=True,
                id_data={"model": "ptm", "data": data},
                scaling_factor=ptm.TransformedVar(
                    value=1.0, name="scaling_factor", prior=scaling_prior
                ),
            )

        if part == "ptm-wb01-bounded":
            logger.warning("STARTING NEW RUN: PTMs with Weibull Prior with Scale 0.01")
            # PTM SDPrior scale 0.01
            run_ptm(
                seed=seed,
                data_path=data_path,
                out_path=out_path,
                warmup=WARMUP,
                posterior=POSTERIOR,
                n=n,
                scale_terms=True,
                sample_transformation=True,
                identifier=f"ptm-wb01-bounded-{data}-N{n}",
                prior_tau2_covariates=IGPRIOR,
                prior_tau2_normalization=SDPRIOR_01,
                cache_path=cache_path,
                optimize_start_values=True,
                skip_if_results_exist=True,
                id_data={"model": "ptm", "data": data},
                scaling_factor=ptm.SymmetricallyBoundedScalar(
                    0.3, name="scaling_factor"
                ),
            )

        if part == "ptm-wb01-tnorm":
            logger.warning("STARTING NEW RUN: PTMs with Weibull Prior with Scale 0.01")
            # PTM SDPrior scale 0.01
            scaling_prior = Dist(
                tfd.TruncatedNormal, loc=1.0, scale=0.1, low=0.01, high=jnp.inf
            )
            run_ptm(
                seed=seed,
                data_path=data_path,
                out_path=out_path,
                warmup=WARMUP,
                posterior=POSTERIOR,
                n=n,
                scale_terms=True,
                sample_transformation=True,
                identifier=f"ptm-wb01-tnorm-{data}-N{n}",
                prior_tau2_covariates=IGPRIOR,
                prior_tau2_normalization=SDPRIOR_01,
                cache_path=cache_path,
                optimize_start_values=True,
                skip_if_results_exist=True,
                id_data={"model": "ptm", "data": data},
                scaling_factor=ptm.TransformedVar(
                    value=1.0, name="scaling_factor", prior=scaling_prior
                ),
            )
        if part == "ptm-wb05-onion-centered":
            logger.warning(
                "STARTING NEW RUN: Onion PTMs with SD Prior, centered, scale = 0.05."
            )
            # PTM SDPrior scale 0.01
            run_ptm_onion(
                seed=seed,
                data_path=data_path,
                out_path=out_path,
                warmup=WARMUP,
                posterior=POSTERIOR,
                n=n,
                scale_terms=True,
                sample_transformation=True,
                identifier=f"ptm-onion-wb05-centered-{data}-N{n}",  # noqa
                prior_tau2_covariates=IGPRIOR,
                prior_tau2_normalization=SDPRIOR_05,
                cache_path=cache_path,
                skip_if_results_exist=True,
                id_data={"model": "ptm", "data": data},
                centered=True,
                scaled=True,
            )
        if part == "ptm-ig-onion-centered":
            logger.warning(
                "STARTING NEW RUN: Onion PTMs with SD Prior, centered, scale = 0.05."
            )
            # PTM SDPrior scale 0.01
            run_ptm_onion(
                seed=seed,
                data_path=data_path,
                out_path=out_path,
                warmup=WARMUP,
                posterior=POSTERIOR,
                n=n,
                scale_terms=True,
                sample_transformation=True,
                identifier=f"ptm-onion-ig-centered-{data}-N{n}",  # noqa
                prior_tau2_covariates=IGPRIOR,
                prior_tau2_normalization=IGPRIOR,
                cache_path=cache_path,
                skip_if_results_exist=True,
                id_data={"model": "ptm", "data": data},
                centered=True,
                scaled=True,
            )

        if part == "ptm-wb05-onion-uncentered":
            logger.warning(
                "STARTING NEW RUN: Onion PTMs with SD Prior, centered, scale = 0.05."
            )
            # PTM SDPrior scale 0.01
            run_ptm_onion(
                seed=seed,
                data_path=data_path,
                out_path=out_path,
                warmup=WARMUP,
                posterior=POSTERIOR,
                n=n,
                scale_terms=True,
                sample_transformation=True,
                identifier=f"ptm-onion-wb05-uncentered-{data}-N{n}",  # noqa
                prior_tau2_covariates=IGPRIOR,
                prior_tau2_normalization=SDPRIOR_05,
                cache_path=cache_path,
                skip_if_results_exist=True,
                id_data={"model": "ptm", "data": data},
                centered=False,
                scaled=False,
            )

        if part == "ptm-ig-onion-uncentered":
            logger.warning(
                "STARTING NEW RUN: Onion PTMs with SD Prior, centered, scale = 0.05."
            )
            # PTM SDPrior scale 0.01
            run_ptm_onion(
                seed=seed,
                data_path=data_path,
                out_path=out_path,
                warmup=WARMUP,
                posterior=POSTERIOR,
                n=n,
                scale_terms=True,
                sample_transformation=True,
                identifier=f"ptm-onion-og-uncentered-{data}-N{n}",  # noqa
                prior_tau2_covariates=IGPRIOR,
                prior_tau2_normalization=IGPRIOR,
                cache_path=cache_path,
                skip_if_results_exist=True,
                id_data={"model": "ptm", "data": data},
                centered=False,
                scaled=False,
            )

        logger.info(f"Finished iteration with {part=}, {n=}, {data=}, {seed=}.")
        # mailog.info(f"Finished iteration with {part=}, {n=}, {data=}, {seed=}.")

        if res is None:
            mailog.info(f"FINISHED: {seed=}, {data=}, {part=}, {n=}")
        else:
            mailog.info(f"SKIPPED: {seed=}, {data=}, {part=}, {n=}")

    logger.info("Finished all iterations.")


if __name__ == "__main__":
    run_simulation()
