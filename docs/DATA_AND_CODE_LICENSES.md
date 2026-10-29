# Data and dependency provenance

M5 raw files are user-provided from the Kaggle competition after acceptance of its rules. They remain in the ignored `data/raw/m5/` directory and are not included in the wheel or interview package. The project does not grant redistribution rights to those files. Reports contain experiment aggregates rather than a copy of the raw competition data.

The implementation uses local CPU libraries and open-source optimization backends. Installed dependency version/license metadata is recorded in `reports/release/dependency_license_metadata.json`; consult the actual upstream package notices for distribution obligations. The installed HiGHS and PyEPO distributions identify MIT licenses, Pyomo identifies BSD-3-Clause, and Streamlit identifies Apache-2.0 in that metadata. This record does not change the terms of any dependency.

No repository-wide license is invented or assigned by the release work. The owner should choose any license before public redistribution. No Kaggle credentials, raw third-party data or proprietary operational records are required in the prepared demonstration bundle.
