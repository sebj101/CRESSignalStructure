from pathlib import Path


def get_dipole_antenna_paths() -> dict[str, Path]:
    """Return paths to the bundled HFSS half-wave dipole antenna CSV files.

    The data files are located at data/antennas/ in the repository root.
    This helper resolves the paths relative to the installed package location,
    so it works from any working directory when the repo has been cloned. For
    your own antenna models a helper function like this is not necessary and one
    can simply use the bare file paths.

    Returns
    -------
    dict with keys 'efield', 'gain', 'impedance' pointing to:
        - EFields.csv     : far-field E-field components (phi/theta grid)
        - GainTotal.csv   : total gain pattern (phi/theta grid)
        - ZParameters.csv : port impedance Z(1,1) vs frequency

    Raises
    ------
    FileNotFoundError
        If the data directory is not found. This occurs when the package was
        pip-installed from a wheel rather than cloned from the repository.

    Notes
    -----
    The pattern_frequency argument for HFSSAntenna should be set to the
    resonant frequency of the antenna, which can be found from the
    zero-crossing of Im(Z) in the impedance data. See
    examples/HFSSvsHalfWaveDipole.py for an example.
    """
    data_dir = Path(__file__).parent.parent / "data" / "antennas"
    if not data_dir.exists():
        raise FileNotFoundError(
            f"Sample data directory not found at {data_dir}. "
            "The HFSS sample data is only available in a git clone of the "
            "CRESSignalStructure repository, not in a pip-installed wheel."
        )
    return {
        "efield": data_dir / "EFields.csv",
        "gain": data_dir / "GainTotal.csv",
        "impedance": data_dir / "ZParameters.csv",
    }