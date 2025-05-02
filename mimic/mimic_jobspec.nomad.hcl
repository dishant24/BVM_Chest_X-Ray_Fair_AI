job "mimic-diagnostic" {

  meta {
    owner = "sutariya"
  }

  datacenters = ["edgecloud"]
  type = "batch"


  group "simple-training-group" {

    # Define host volumes with proper read/write permissions
    volume "input" {
      type      = "csi"
      read_only = true
      source    = "default_mimic_cxr"
      attachment_mode = "file-system"
      access_mode = "multi-node-multi-writer"
    }
    volume "output_folder" {
      type      = "csi"
      read_only = false
      source    = "default_dl_output"
      attachment_mode = "file-system"
      access_mode = "multi-node-multi-writer"
    }
    
  

    task "mimic-diagnostic-training" {
      leader = true
      driver = "docker"
      config {
        image = "registry.fme.lan/dockerhub/pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime"
        command = "bash"
        args = ["-c", "pip install pandas numpy matplotlib torchvision torch tqdm scikit-image wandb torchcontrib scikit-learn seaborn scikit-learn --root-user-action=ignore && export WANDB_API_KEY=c97efa068ce628aa2d4ad9bbc8b2b2dbaa6c6387 && wandb login && python deep_learning/output/Sutariya/main/mimic/mimic_cxr_model.py"]
        work_dir = "/"
        shm_size = 17179869184
      }

      # Mount the volumes correctly
      volume_mount {
        volume = "input"
        destination = "/MIMIC-CXR"
        read_only = true
      }
      volume_mount {
        volume = "output_folder"
        destination = "/deep_learning/output"
        read_only = false
      }

      resources {
        cpu = 8000
        memory = 30000
        device "nvidia/gpu" {
          count = 1
        }
      }
    }

    restart {
      attempts = 2
      mode = "fail"
    }
    
    reschedule {
      attempts  = 0
      unlimited = false
    }
  }
}
